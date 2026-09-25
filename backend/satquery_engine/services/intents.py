"""Compositional intent grammar and allow-listed execution DAG.

Rules match concepts, synonyms and request operators, not full query strings.
Unknown requests abstain; a future language parser must emit this same schema.
"""
from enum import Enum
import re
from satquery_engine.schemas import PlanNode, TaskPlan, TaskType

Intent = Enum("Intent", {name: name for name in '''SINGLE_VQA SCENE_DESCRIPTION GROUNDING BUILDING_COUNT BUILDING_DETECTION BUILDING_FOOTPRINT BUILDING_GROUNDING BUILDING_CHANGE NEW_BUILDING_COUNT OBJECT_COUNT OBJECT_GROUNDING WATER_ANALYSIS WATER_GROUNDING WATER_AREA VEGETATION_ANALYSIS VEGETATION_CHANGE BUILT_UP_ANALYSIS BUILT_UP_CHANGE LAND_COVER_ANALYSIS CHANGE_DETECTION CHANGE_DESCRIPTION CHANGE_VQA CHANGE_AREA CHANGE_GROUNDING NEW_OBJECT_DETECTION REMOVED_OBJECT_DETECTION OPTICAL_SAR_ANALYSIS OPTICAL_SAR_COMPARISON OPTICAL_SAR_FUSION MULTI_INTENT UNSUPPORTED_TASK INVALID_REQUEST'''.split()}, type=str)
ALLOWED_TOOLS = {"validate", "register", "buildings", "building_match", "spectral", "spectral_change", "land_cover", "change", "fusion", "vlm", "synthesize", "report"}


def has(q, pattern):
    return bool(re.search(r"\b(?:" + pattern + r")\b", q))


def plan_query(query: str, image_count: int = 1, pair_type: str = "auto", configuration: str | None = None) -> TaskPlan:
    q = query.lower().strip().replace("-", " ")
    if not q:
        raise ValueError("Query cannot be empty")
    building = has(q, r"buildings?|houses?|homes?|structures?|rooftops?|footprints?")
    water = has(q, r"waters?|ndwi|rivers?|lakes?|ponds?|reservoirs?|oceans?|seas?|flood(?:s|ed|ing)?|canals?")
    veg = has(q, r"vegetation|forests?|trees?|crops?|greenery|canopy|ndvi|deforestation")
    built = has(q, r"built up|urban|construction|settlements?|cities|city|towns?|ndbi")
    other_object = has(q, r"roads?|bridges?|vehicles?|cars?|ships?|aircraft|airplanes?|airports?|containers?|tanks?")
    count = has(q, r"count|counting|how many|number|tally|total|enumerate")
    ground = has(q, r"where|find|locate|show|mark|highlight|outline|delineate|detect|identify|map")
    area = has(q, r"area|how much|size|extent|coverage|hectares|percentage|percent")
    change = has(q, r"chang(?:e|ed|es|ing)|increas(?:e|ed)|decreas(?:e|ed)|more|fewer|new|removed|disappeared|appeared|loss|lost|expanded|expansion|became|growth|grew|shrank|decreased|before|after|between|now")
    compare = has(q, r"compare|comparison|difference|different")
    fusion = has(q, r"sar|radar") and (has(q, r"optical|combine|fusion|fuse|together|compare|both") or image_count == 2)
    fusion = fusion or has(q, "both sensors|cross modal") or pair_type == "optical_sar"
    fusion = fusion or (compare and configuration in {"PAIR_OPTICAL_SAR", "PAIR_MULTISENSOR_BITEMPORAL"})
    change = (change or compare) and not fusion
    scene = has(q, r"describe|description|overview|summarize|caption|what is happening|what do you see|what can you see|what does .+ contain")
    land = has(q, r"land cover|landcover|land use|terrain|land types?|types? of land|land is visible|land")
    comprehensive = has(q,r"completely|comprehensive|everything|full analysis|analy[sz]e (?:this |the )?(?:image|scene) fully")
    if comprehensive:
        building=water=veg=built=scene=land=True
        count=True
        change=image_count==2 and configuration in {"PAIR_BITEMPORAL_OPTICAL","PAIR_BITEMPORAL_SAR"}
        fusion=configuration in {"PAIR_OPTICAL_SAR","PAIR_MULTISENSOR_BITEMPORAL"}
    if land and not change and not fusion:
        water = veg = True
    unsupported = has(q, r"capital of|president|who is|who was|write (?:a |some )?(?:code|script|poem)|python code|recipe|cook|translate|stock price|weather forecast|meaning of life")
    intents = []
    nodes = [PlanNode(node_id="validate", tool="validate")]
    def add_intent(value):
        if value not in intents:
            intents.append(value)
    def node(key, tool, deps=None, **params):
        nodes.append(PlanNode(node_id=key, tool=tool, depends_on=deps or ["validate"], parameters=params))
    if unsupported or not any([building, water, veg, built, scene, land, change, fusion, ground, other_object]):
        return TaskPlan(task=TaskType.UNSUPPORTED if unsupported else TaskType.UNCLEAR,
                        application="unsupported" if unsupported else "unclear", tools=[], reason="No supported remote-sensing request was established.",
                        intents=["UNSUPPORTED_TASK" if unsupported else "INVALID_REQUEST"],
                        clarification=None if unsupported else "What would you like to find out about the uploaded image?")
    if change:
        add_intent("CHANGE_DETECTION")
        node("register", "register")
    deps = ["register"] if change else ["validate"]
    if fusion:
        add_intent("OPTICAL_SAR_FUSION")
        node("register", "register")
        node("fusion", "fusion", ["register"], target="water" if water else None)
    elif building:
        add_intent("BUILDING_COUNT" if count else "BUILDING_DETECTION")
        if ground:
            add_intent("BUILDING_GROUNDING")
        if area:
            add_intent("BUILDING_FOOTPRINT")
        node("buildings_a", "buildings", deps, asset=0)
        if change:
            add_intent("BUILDING_CHANGE")
            if has(q, "new|appeared") and count:
                add_intent("NEW_BUILDING_COUNT")
            node("buildings_b", "buildings", deps, asset=1)
            node("building_match", "building_match", ["buildings_a", "buildings_b"])
    if not fusion:
        for present, target, base in [(water,"water","WATER"),(veg,"vegetation","VEGETATION"),(built,"built-up","BUILT_UP")]:
            if not present:
                continue
            if target == "built-up" and building:
                add_intent("BUILT_UP_CHANGE" if change else "BUILT_UP_ANALYSIS")
                continue  # Reuse measured building footprints, rather than run an unrelated index.
            add_intent(base + ("_CHANGE" if change and base != "WATER" else "_ANALYSIS"))
            if water and target == "water":
                if ground: add_intent("WATER_GROUNDING")
                if area: add_intent("WATER_AREA")
            node(target.replace("-", "_") + "_measure", "spectral_change" if change else "spectral", deps,
                 target=target, largest=has(q, "largest|biggest"), strict=has(q, "ndvi|ndwi|ndbi"))
        if change and not any([building, water, veg, built]):
            add_intent("CHANGE_AREA" if area else "CHANGE_DESCRIPTION")
            node("change", "change", deps)
        if land and not change:
            add_intent("LAND_COVER_ANALYSIS")
            node("land_cover", "land_cover", ["validate"])
        if scene or other_object or (ground and not any([building, water, veg, built, land, change])):
            add_intent("LAND_COVER_ANALYSIS" if land else "SCENE_DESCRIPTION" if scene else "OBJECT_COUNT" if count else "OBJECT_GROUNDING")
            node("scene", "vlm", deps, grounding=other_object or (ground and not (scene or land)), counting=count and other_object)
    leaves = [n.node_id for n in nodes if n.tool not in {"validate", "register"}]
    node("synthesize", "synthesize", leaves or ["validate"])
    node("report", "report", ["synthesize"])
    task = TaskType.OPTICAL_SAR if fusion else TaskType.BUILDING_CHANGE if change and building else TaskType.BI_TEMPORAL_CHANGE if change else TaskType.BUILDING_COUNT if building else TaskType.LAND_COVER if land else TaskType.WATER_ANALYSIS if water else TaskType.VEGETATION_ANALYSIS if veg else TaskType.BUILT_UP_ANALYSIS if built else TaskType.LAND_COVER if land else TaskType.SCENE_DESCRIPTION if scene else TaskType.OBJECT_IDENTIFICATION
    return TaskPlan(task=task, application="optical_sar" if fusion else "bi_temporal" if change else "single_image",
                    specific_task=task.value, intents=intents, sub_tasks=intents, multi_intent=len(intents)>1,
                    target="built-up" if building or built else "water" if water else "vegetation" if veg else None,
                    asks_direction=change, requires_pair=change or fusion, requires_temporal_relationship=change,
                    tools=list(dict.fromkeys(n.tool for n in nodes)), nodes=nodes,
                    reason="Composed requested objects, measurements and temporal/sensor relationships into a validated tool DAG.")


def validate_dag(plan):
    seen = set()
    tools = {}
    if plan.nodes and (plan.nodes[0].tool!="validate" or plan.nodes[0].depends_on):
        raise ValueError("Plan must validate inputs before executing any tool")
    parameter_keys={"validate":set(),"register":set(),"buildings":{"asset"},"building_match":set(),
        "spectral":{"target","largest","strict"},"spectral_change":{"target","largest","strict"},
        "change":set(),"land_cover":set(),"fusion":{"target"},"vlm":{"grounding","counting"},"synthesize":set(),"report":set()}
    for node in plan.nodes:
        if node.tool not in ALLOWED_TOOLS or node.node_id in seen or any(d not in seen for d in node.depends_on):
            raise ValueError("Invalid or non-allow-listed execution DAG")
        if set(node.parameters)-parameter_keys[node.tool]: raise ValueError("Unsupported tool parameters")
        if node.tool!="validate" and not node.depends_on: raise ValueError("Every tool requires validated input dependencies")
        if node.tool=="buildings" and (type(node.parameters.get("asset")) is not int or node.parameters["asset"] not in (0,1)):
            raise ValueError("Invalid building asset index")
        if node.tool in {"spectral","spectral_change"} and node.parameters.get("target") not in {"water","vegetation","built-up"}:
            raise ValueError("Unsupported spectral target")
        for key in ("largest","strict","grounding","counting"):
            if key in node.parameters and type(node.parameters[key]) is not bool: raise ValueError("Tool flags must be boolean")
        if node.tool in {"change","spectral_change","fusion"} and not any(tools.get(d)=="register" for d in node.depends_on):
            raise ValueError("Paired tools require registration")
        if node.tool=="building_match" and (len(node.depends_on)!=2 or any(tools.get(d)!="buildings" for d in node.depends_on)):
            raise ValueError("Building comparison requires two building outputs")
        if node.tool=="report" and not any(tools.get(d)=="synthesize" for d in node.depends_on):
            raise ValueError("Report requires canonical synthesis")
        seen.add(node.node_id)
        tools[node.node_id]=node.tool
