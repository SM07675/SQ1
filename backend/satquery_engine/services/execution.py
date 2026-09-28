"""Validated DAG executor using the existing specialist/GIS architecture."""
import asyncio
import json
import time
from datetime import datetime, UTC
from pathlib import Path
import numpy as np
from scipy import ndimage
from satquery_engine.schemas import AnalysisResponse, ArtifactRef, EvidenceItem, GeoVerdict, TraceStep, VerdictStatus, TaskType, TaskPlan, PlanNode
from satquery_engine.services.intents import plan_query, validate_dag
from satquery_engine.services.input_configuration import classify_configuration, policy_blockers, node_policy_error, acquisition_time
from satquery_engine.services.raster import validate_inputs, render_preview
from satquery_engine.services.alignment import align_pair
from satquery_engine.services.buildings import detect_buildings, match_buildings
from satquery_engine.services.measurements import measure_cover, measure_cover_change
from satquery_engine.services.model_registry import registry
from satquery_engine.services.confidence import confidence_breakdown
from satquery_engine.services.report import write_pdf_report, write_manifest
from satquery_engine.services.geolocation import parse_user_coordinates


async def analyze(*,result_id,query,pair_type,image_paths,output_dir,progress=None,input_filenames=None,input_asset_ids=None):
    started=time.perf_counter(); output_dir.mkdir(parents=True,exist_ok=True)
    trace=[]; artifacts=[]; evidence=[]; findings=[]; statistics={}; models=[]; limitations=[]; results={}
    def emit(message):
        if progress: progress(message)
    def add_path(path):
        rel=path.relative_to(output_dir).as_posix()
        mime={".png":"image/png",".tif":"image/tiff",".geojson":"application/geo+json",".pdf":"application/pdf",".json":"application/json"}.get(path.suffix,"application/octet-stream")
        if not any(a.url.endswith('/'+rel) for a in artifacts):
            name = path.name if not any(a.name == path.name for a in artifacts) else rel.replace('/', '_')
            import uuid
            artifacts.append(ArtifactRef(artifact_id=str(uuid.uuid5(uuid.NAMESPACE_URL,f"{result_id}/{rel}")),name=name,url=f"/artifacts/{result_id}/{rel}",mime_type=mime))
    emit("Reading and checking image metadata")
    metadata,quality=await asyncio.to_thread(validate_inputs,image_paths)
    if input_filenames:
        metadata = [asset.model_copy(update={"filename": name}) for asset, name in zip(metadata, input_filenames)]
    if input_asset_ids:
        metadata=[asset.model_copy(update={"asset_id":ident}) for asset,ident in zip(metadata,input_asset_ids,strict=True)]
    configuration,clarification=classify_configuration(metadata,pair_type)
    emit("Understanding your request and planning analysis")
    user_coordinates=parse_user_coordinates(query)
    plan=(TaskPlan(task=TaskType.GROUNDING,application="user_location",specific_task="user_location",
                   intents=["USER_LOCATION"],tools=["validate","synthesize","report"],
                   reason="Validated explicit user coordinates without model inference.",
                   nodes=[PlanNode(node_id="validate",tool="validate"),
                          PlanNode(node_id="synthesize",tool="synthesize",depends_on=["validate"]),
                          PlanNode(node_id="report",tool="report",depends_on=["synthesize"])])
          if user_coordinates is not None else plan_query(query,len(image_paths),pair_type,configuration))
    validate_dag(plan)
    clarification=plan.clarification or (clarification if len(image_paths)==2 else None)
    if plan.requires_temporal_relationship and configuration.startswith("PAIR_BITEMPORAL"):
        dates=[a.acquisition_date for a in metadata]
        if all(dates):
            try:
                order=sorted(range(2),key=lambda i: acquisition_time(dates[i]))
                image_paths=[image_paths[i] for i in order]; metadata=[metadata[i] for i in order]
            except ValueError: pass
    quality.blockers.extend(policy_blockers(plan,metadata,configuration))
    quality.compatible=not quality.blockers and not clarification
    trace.append(TraceStep(step=1,component="validation_and_planner",action="Checked metadata, pair semantics and allow-listed task DAG",
                           status="ok" if quality.compatible else "blocked",duration_ms=round((time.perf_counter()-started)*1000),
                           details={"configuration":configuration,"intents":plan.intents,"blockers":quality.blockers}))
    for i,path in enumerate(image_paths):
        preview=output_dir/f"preview_{i+1}.png"
        await asyncio.to_thread(render_preview,path,preview)
        add_path(preview)
    if user_coordinates is not None and quality.compatible:
        latitude,longitude=user_coordinates
        location={"type":"user_location","latitude":latitude,"longitude":longitude,"source":"USER_COORDINATES"}
        statistics["user_location"]=location
        summary=f"Marked the user-provided location at latitude {latitude}, longitude {longitude}."
        point_file=output_dir/"user_location.geojson"
        point_file.write_text(json.dumps({"type":"FeatureCollection",
            "properties":{"crs":"EPSG:4326","coordinate_space":"geographic","source":"USER_COORDINATES"},
            "features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[longitude,latitude]},
                         "properties":location}]}),encoding="utf-8")
        add_path(point_file)
        evidence.append(EvidenceItem(evidence_id="evidence_1",kind="user_location",producer="user",
                                     summary=summary,confidence=1.0,confidence_source="authoritative_user_input",
                                     artifact_url=artifacts[-1].url,metrics=location))
        findings.append({"text":summary,"evidence_ids":["evidence_1"],"status":"supported"})
        trace.append(TraceStep(step=len(trace)+1,component="user_location",action="Validated and mapped explicit coordinates",
                               status="ok",duration_ms=0,details={"source":"USER_COORDINATES"}))
    working=list(image_paths); completed={"validate"}; failed=set()
    from satquery_engine.services.surface_context import SurfaceContext
    surfaces = SurfaceContext(output_dir / "surface_context")
    preflight={n.node_id:node_policy_error(n,metadata) for n in plan.nodes}
    if quality.compatible and plan.task not in {TaskType.UNSUPPORTED,TaskType.UNCLEAR}:
        for node in plan.nodes:
            if node.tool in {"validate","synthesize","report"}: continue
            if any(d in failed for d in node.depends_on):
                failed.add(node.node_id)
                trace.append(TraceStep(step=len(trace)+1,component=node.tool,action="Skipped because required evidence is unavailable",status="skipped",duration_ms=0))
                continue
            tick=time.perf_counter(); emit({"register":"Aligning images","buildings":"Running building detection","building_match":"Comparing building footprints", "spectral":"Measuring image features","spectral_change":"Measuring change","land_cover":"Mapping land cover from image evidence","change":"Measuring image differences","fusion":"Checking optical and radar evidence","vlm":"Describing image evidence"}[node.tool])
            sub=output_dir/node.node_id; sub.mkdir(exist_ok=True)
            params=node.parameters
            try:
                if preflight.get(node.node_id): raise ValueError(preflight[node.node_id])
                result=None; summary=""
                if node.tool=="register":
                    a,b,registration=await asyncio.to_thread(align_pair,*working,output_dir,not plan.requires_temporal_relationship)
                    working=[a,b]; add_path(output_dir/"registration_report.json")
                    quality.checks["alignment_score"]=registration.alignment_score
                    quality.checks["registration_method"]=registration.method
                    if plan.requires_temporal_relationship and registration.alignment_score<.35:
                        raise ValueError("The images could not be aligned reliably enough for change analysis.")
                    limitations.append(registration.message)
                elif node.tool=="buildings":
                    emit("Checking water boundaries before counting buildings")
                    water_context = await asyncio.to_thread(surfaces.water, working[params["asset"]])
                    for path in water_context.get("paths", []): add_path(path)
                    models.extend(water_context.get("models_used", []))
                    result=await asyncio.to_thread(detect_buildings,working[params["asset"]],sub,emit,water_result=water_context)
                    summary=f'I found approximately {result["count"]} visible building footprints in image {params["asset"]+1}.'
                    summary += f' Their masks cover {result["selected_pixels"]} of {result["valid_pixels"]} valid image pixels ({result["coverage_percent"]:.2f}%).'
                    if result.get("count_reliability") == "COUNT_UNRELIABLE":
                        summary += ' This is a low-confidence detection count; an accurate building inventory cannot be established from this result.'
                    if result["count"] == 0:
                        summary += ' No footprints were detected; this does not establish that no buildings are present.'
                    if any(i in plan.intents for i in ["BUILT_UP_ANALYSIS", "BUILT_UP_CHANGE", "BUILDING_FOOTPRINT"]):
                        summary += f' Their measured footprint area is {result["area_m2"]:.1f} square metres.' if result["area_m2"] is not None else f' They cover {result["selected_pixels"]} image pixels; geographic area is unknown.'
                    models.append(result["model_id"])
                elif node.tool=="building_match":
                    result=await asyncio.to_thread(match_buildings,results["buildings_a"],results["buildings_b"],sub)
                    summary=f'The detected building count changed from {result["before_count"]} to {result["after_count"]}. There are {result["possible_new_count"]} possible new and {result["possible_removed_count"]} possibly removed footprints.'
                    if result["net_count_change_percent"] is not None:
                        summary += f' Net detected count change is {result["net_count_change_percent"]:+.2f}% relative to the first image.'
                    if result['net_footprint_area_m2'] is not None:
                        summary += f' The net detected footprint area change is {result["net_footprint_area_m2"]:+.1f} square metres; this measures buildings, not all built-up surfaces.'
                elif node.tool=="spectral":
                    if params["target"] == "water":
                        result=await asyncio.to_thread(surfaces.water,working[0],largest=params.get("largest",False),strict=params.get("strict",False))
                    else:
                        result=await asyncio.to_thread(measure_cover,working[0],sub,**params)
                    unit=f'{result["area_m2"]:.1f} square metres' if result["area_m2"] is not None else f'{result["selected_pixels"]} sampled image pixels'
                    loc=f' in the {result["location_description"]}' if result.get("location_description") else ''
                    if params["target"] == "water" and result.get("evidence_state") == "INSUFFICIENT_EVIDENCE":
                        summary="Water boundaries could not be verified from this RGB image because the compatible aerial models failed."
                    else:
                        summary=f'{result["method"]} identifies {params["target"]}-like regions{loc} covering {result["selected_pixels"]} of {result["valid_pixels"]} valid image pixels ({result["coverage_percent"]:.2f}%; {unit}).'
                elif node.tool=="land_cover":
                    from satquery_engine.services.land_cover import classify_land_cover_composite
                    result=await asyncio.to_thread(classify_land_cover_composite,working[0],sub,query,
                        water_result=await asyncio.to_thread(surfaces.water,working[0]), building_result=results.get("buildings_a"),
                        vegetation_result=results.get("vegetation_measure"))
                    summary=result["summary"]
                elif node.tool=="spectral_change":
                    result=await asyncio.to_thread(measure_cover_change,*working,sub,params["target"])
                    summary=(f'The {params["target"]} index threshold covers {result["before_pixels"]} of {result["valid_pixels"]} shared valid pixels '
                             f'({result["before_percent"]:.2f}%) before and {result["after_pixels"]} ({result["after_percent"]:.2f}%) after. '
                             f'Gain is {result["gain_pixels"]} pixels ({result["gain_percent"]:.2f}%); loss is '
                             f'{result["loss_pixels"]} pixels ({result["loss_percent"]:.2f}%). '
                             f'Net change is {result["net_percentage_points"]:+.2f} percentage points.')
                elif node.tool=="change":
                    from satquery_engine.services.raster import deterministic_change_detection
                    result=await asyncio.to_thread(deterministic_change_detection,*working,sub)
                    result.pop("mask_path"); result.pop("geojson_path")
                    result["method"]="Radiometric difference threshold; not a learned change model"
                    result["evidence_strength"]=0.0
                    result["limitations"]=["This measures image differences, which can reflect illumination or season. The learned change specialist is unavailable; the type of change is unverified."]
                    summary=f'I measured image differences in {result["changed_pixels"]} of {result["total_pixels"]} shared valid pixels ({result["changed_percent"]:.2f}%) at native resolution. I cannot establish what caused them.'
                elif node.tool=="fusion":
                    external=await registry.invoke("croma",{"optical_path":str(working[0]),"sar_path":str(working[1])})
                    if not external.get("available"): raise ValueError("The optical-SAR fusion specialist is unavailable. No fused result was generated.")
                    # A representation alone cannot ground semantic spatial claims.
                    raise ValueError("The fusion service returned a representation, but a validated task-specific decoder is not configured. No geographic fusion claim was generated.")
                elif node.tool=="vlm":
                    if params.get("counting"):
                        raise ValueError("A dedicated detector for this object type is unavailable. A language model will not supply an exact count.")
                    facts = {name:{k:v for k,v in value.items() if k in {"count","coverage_percent","area_m2","breakdown","method","limitations"}} for name,value in results.items()}
                    explanation_query = query + "\nExplain only the following backend evidence. Do not invent or revise counts, areas, coordinates, confidence or percentages. Do not introduce numerical claims in the scene description. Backend facts: " + json.dumps(facts,allow_nan=False)
                    external=await registry.invoke_vlm(query=explanation_query,image_paths=[str(p) for p in working],tile_paths=[],metadata=[a.model_dump(mode="json") for a in metadata])
                    if not external.get("available"): raise ValueError("The image description model is unavailable. No description was generated.")
                    if params.get("grounding"):
                        raise ValueError("A validated spatial detector for this object is unavailable; text alone cannot establish its location.")
                    summary=external.get("answer","")
                    if not summary: raise ValueError("The image description model returned no answer.")
                    # Numerical measurements never come from language output.
                    import re
                    if re.search(r"\d",summary): raise ValueError("The description included unsupported numerical claims and was withheld.")
                    models.append(external["model"])
                    result={"method":external["model"],"evidence_strength":0.0,"paths":[],
                            "limitations":["This description is a model interpretation; no exact geographic measurements were made."]}
                if result is not None:
                    from satquery_engine.services.quality_gate import validate_result_evidence
                    validate_result_evidence(result,output_dir / "surface_context" if node.tool == "spectral" and params["target"] == "water" else sub)
                    results[node.node_id]=result
                    models.extend(result.get("models_used", []))
                    for path in result.get("paths",[]): add_path(path)
                    metrics={k:v for k,v in result.items() if k not in {"paths","features","limitations","mask_path","geojson_path"} and not isinstance(v,np.ndarray)}
                    statistics[node.node_id]=metrics
                    eid=f"evidence_{len(evidence)+1}"
                    evidence.append(EvidenceItem(evidence_id=eid,kind=node.tool,producer=result.get("model_id",result.get("method",node.tool)),summary=summary,
                                  confidence=float(result.get("evidence_strength",0)),confidence_source="uncalibrated_model_score" if node.tool=="buildings" else "documented_measurement_strength",
                                  artifact_url=next((a.url for a in artifacts if any(a.url.endswith('/'+p.relative_to(output_dir).as_posix()) for p in result.get("paths",[]) if 'overlay' in p.name)),None),metrics=metrics,
                                  timestamp=datetime.now(UTC).isoformat(),asset_indices=[params["asset"]] if "asset" in params else list(range(len(working)))))
                    findings.append({"text":summary,"evidence_ids":[eid],"status":"supported_with_limitations"})
                    limitations.extend(result.get("limitations",[]))
                completed.add(node.node_id)
                trace.append(TraceStep(step=len(trace)+1,component=node.tool,action=node.node_id,status="ok",duration_ms=round((time.perf_counter()-tick)*1000),details={"preprocessing":result.get("preprocessing") if result else None,"checkpoint_sha256":result.get("checkpoint_sha256") if result else None}))
            except (ValueError,ImportError,RuntimeError,OSError) as exc:
                failed.add(node.node_id); limitations.append(str(exc))
                findings.append({"text":str(exc),"evidence_ids":[],"status":"insufficient_evidence"})
                trace.append(TraceStep(step=len(trace)+1,component=node.tool,action=node.node_id,status="unavailable",duration_ms=round((time.perf_counter()-tick)*1000),details={"message":str(exc)}))
    emit("Collecting evidence and preparing your answer")
    limitations=list(dict.fromkeys(limitations+quality.warnings))
    if clarification:
        answer=clarification; status=VerdictStatus.INSUFFICIENT_EVIDENCE
    elif quality.blockers:
        answer=" ".join(quality.blockers); status=VerdictStatus.INVALID_INPUT
    elif plan.task==TaskType.UNSUPPORTED:
        answer="This request is outside the supported satellite-image analysis tasks."; status=VerdictStatus.UNSUPPORTED_TASK
    else:
        answer=" ".join(f["text"] for f in findings) or "No evidence was produced for this request."
        status=(VerdictStatus.SUPPORTED if user_coordinates is not None and evidence else
                VerdictStatus.SUPPORTED_WITH_LIMITATIONS if evidence else VerdictStatus.INSUFFICIENT_EVIDENCE)
        if any(r.get("count_reliability") == "COUNT_UNRELIABLE" or
               (r.get("benchmark_f1") is not None and r["benchmark_f1"]<.6) for r in results.values()):
            status=VerdictStatus.LOW_CONFIDENCE
        if any((r.get("aerial_water_agreement") or {}).get("withheld_fraction", 0) > .5
               for r in results.values()):
            status=VerdictStatus.LOW_CONFIDENCE
        if any(r.get("evidence_state") == "INSUFFICIENT_EVIDENCE" for r in results.values()):
            status=(VerdictStatus.INSUFFICIENT_EVIDENCE if plan.task == TaskType.WATER_ANALYSIS
                    else VerdictStatus.LOW_CONFIDENCE)
    breakdown=confidence_breakdown(quality,evidence)
    verdict=GeoVerdict(status=status,answer=answer,confidence=breakdown.final_score,
                       confidence_kind="authoritative_user_input" if user_coordinates is not None and evidence else "uncalibrated_evidence_strength",
                       limitations=limitations,confidence_breakdown=breakdown)
    # List canonical downloadable outputs before serialization, so JSON/PDF/API agree.
    add_path(output_dir/"GeoProof_Report.pdf"); add_path(output_dir/"analysis.json")
    response=AnalysisResponse(result_id=result_id,query=query,generated_at=datetime.now(UTC).isoformat(),task_plan=plan,
              mode="automatic_evidence_analysis",assets=metadata,quality=quality,evidence=evidence,verdict=verdict,trace=trace,
              artifacts=artifacts,input_configuration=configuration,findings=findings,statistics=statistics,models_used=list(dict.fromkeys(models)),
              clarification=clarification,report_id=result_id,timings={"analysis_ms":round((time.perf_counter()-started)*1000)})
    emit("Creating your report")
    await asyncio.to_thread(write_pdf_report,output_dir,response.model_dump(mode="json"))
    await asyncio.to_thread(write_manifest,output_dir,response.model_dump(mode="json"))
    return response
