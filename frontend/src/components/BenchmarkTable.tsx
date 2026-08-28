import React from "react";
import type { BenchmarkItemResult } from "../types";
import { StatusPill } from "./shared/StatusPill";

interface BenchmarkTableProps {
  results: BenchmarkItemResult[];
  datasetName: string;
}

export const BenchmarkTable: React.FC<BenchmarkTableProps> = ({
  results,
  datasetName,
}) => {
  return (
    <div className="bench-table-card">
      <div className="table-header-bar">
        <div className="table-title-group">
          <h3>Frozen-Split Evaluation Ledger</h3>
          <span className="dataset-tag">{datasetName}</span>
        </div>
        <span className="table-count-badge">{results.length} evaluated items</span>
      </div>

      <div className="table-scroll-wrapper">
        <table className="bench-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Sample ID</th>
              <th>Category</th>
              <th>Question</th>
              <th>Predicted vs Expected</th>
              <th>Similarity</th>
              <th>IoU</th>
              <th>Confidence</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            {results.map((item) => (
              <tr key={item.sample_id} className={item.passed ? "row-pass" : "row-fail"}>
                <td>
                  <StatusPill
                    variant={item.passed ? "passed" : "failed"}
                    label={item.passed ? "PASS" : "FAIL"}
                  />
                </td>
                <td>
                  <code className="sample-id-code">{item.sample_id}</code>
                </td>
                <td>
                  <span className="cat-badge">{item.category}</span>
                </td>
                <td className="question-cell">{item.question}</td>
                <td>
                  <div className="pred-vs-exp-box">
                    <div className="pred-line">
                      <span className="pe-tag pred">Pred:</span>
                      <span className="pe-val">{item.predicted_answer}</span>
                    </div>
                    <div className="exp-line">
                      <span className="pe-tag exp">Exp:</span>
                      <span className="pe-val">{item.expected_answer}</span>
                    </div>
                  </div>
                </td>
                <td className="metric-cell font-mono">
                  {(item.semantic_similarity * 100).toFixed(1)}%
                </td>
                <td className="metric-cell font-mono">
                  {typeof item.iou === "number" ? `${(item.iou * 100).toFixed(1)}%` : "—"}
                </td>
                <td className="metric-cell font-mono">
                  {(item.confidence * 100).toFixed(1)}%
                </td>
                <td className="metric-cell font-mono">{item.latency_ms} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
