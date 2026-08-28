CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS analysis_runs (
  result_id UUID PRIMARY KEY,
  task_type TEXT NOT NULL,
  query_text TEXT NOT NULL,
  verdict_status TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  manifest JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evidence_geometries (
  evidence_id BIGSERIAL PRIMARY KEY,
  result_id UUID NOT NULL REFERENCES analysis_runs(result_id) ON DELETE CASCADE,
  producer TEXT NOT NULL,
  evidence_kind TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  geometry GEOMETRY(GEOMETRY, 4326),
  properties JSONB NOT NULL DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS evidence_geometry_gix
  ON evidence_geometries USING GIST (geometry);

