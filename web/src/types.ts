// Result JSON schema v1 — mirrors engine/src/pitchcoach/models.py (keys stay snake_case).

export type Severity = "info" | "warn" | "error";
export type Confidence = "high" | "medium" | "low";
export type Hand = "R" | "L";
export type CameraSide = "arm" | "glove";
export type SegmentName = "legs" | "pelvis" | "trunk" | "arm";

export interface VideoMeta {
  /** Container frame rate: maps frame index <-> playback time. */
  fps: number;
  width: number;
  height: number;
  n_frames: number;
  duration_s: number;
  /**
   * Frame rate of the motion (the capture fps of a slow-motion export; else the container fps):
   * converts frame counts to milliseconds. null only when the file gave no usable fps.
   */
  capture_fps: number | null;
  /** Frame rate of the norms the metrics were scored against; null when nothing was scored. */
  norm_fps: number | null;
}

export interface Subject {
  height_cm: number;
  hand: Hand;
  camera_side: CameraSide;
}

export interface AnalysisWarning {
  code: string;
  severity: Severity;
  message: string;
}

/** Joint pixel coordinates per frame: [x, y, visibility]; null when not finite. */
export type KeypointFrame = Array<Array<number | null>>;

export interface Keypoints {
  joints: string[];
  frames: KeypointFrame[];
}

export interface Events {
  leg_lift_peak: number | null;
  foot_plant: number | null;
  mer_est: number | null;
  release: number | null;
}

export interface Phase {
  name: string;
  label: string;
  start: number;
  end: number;
}

export interface Norm {
  mean: number;
  sd: number;
  direction: "higher" | "lower" | "target";
  source: string;
}

export interface Metric {
  name: string;
  label: string;
  value: number | null;
  unit: string;
  confidence: Confidence;
  segment: string;
  score: number | null;
  z: number | null;
  norm: Norm | null;
}

export interface SegmentScore {
  segment: string;
  label: string;
  score: number | null;
  confidence: Confidence;
}

export interface Leak {
  segment: string;
  frame: number;
  kind: "out_of_order" | "speed_drop";
  severity: number;
  message: string;
}

export interface EnergyPeaks {
  pelvis_center: number | null;
  shoulder_center: number | null;
  elbow: number | null;
  wrist: number | null;
}

export interface Energy {
  segments: string[];
  /** Per frame, one value per segment (0..1). */
  intensity: number[][];
  leaks: Leak[];
  peaks: EnergyPeaks;
}

export interface SelfTestOption {
  id: string;
  label: string;
}

export interface SelfTest {
  id: string;
  instruction: string;
  options: SelfTestOption[];
  implies_cause_on: string;
  source: string;
}

export interface Exercise {
  id: string;
  name: string;
  sets_reps: string;
  description: string;
  source: string;
}

export interface Cause {
  id: string;
  title: string;
  self_test: SelfTest;
  exercises: Exercise[];
}

export interface Finding {
  issue_id: string;
  title: string;
  evidence: string;
  causes: Cause[];
}

export interface AnalysisResult {
  schema_version: number;
  meta: VideoMeta;
  subject: Subject;
  warnings: AnalysisWarning[];
  throw_direction: number;
  keypoints: Keypoints;
  events: Events;
  phases: Phase[];
  metrics: Metric[];
  segments: SegmentScore[];
  energy: Energy;
  findings: Finding[];
}

export type AnalysisStatus = "processing" | "done" | "failed";

export interface AnalysisState {
  status: AnalysisStatus;
  error: string | null;
  result: AnalysisResult | null;
}

export interface AnalysisSummary {
  id: string;
  created_at: string;
  status: AnalysisStatus;
}
