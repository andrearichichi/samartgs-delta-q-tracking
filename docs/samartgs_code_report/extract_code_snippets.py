#!/usr/bin/env python3
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPORT_ROOT / "assets" / "snippets"


class Snippet:
    def __init__(self, slug, title, file, start_line, end_line, purpose, section, language):
        self.slug = slug
        self.title = title
        self.file = file
        self.start_line = start_line
        self.end_line = end_line
        self.purpose = purpose
        self.section = section
        self.language = language

    def to_json(self):
        return {
            "slug": self.slug,
            "title": self.title,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "purpose": self.purpose,
            "section": self.section,
            "language": self.language,
            "output": "assets/snippets/%s.txt" % self.slug,
        }


SNIPPETS = [
    Snippet("config_usb", "Configurazione USB", "scripts/delta_q_tracking/config_usb.yaml", 1, 48, "Mostra path, trajectory, moving part, optimizer e loss.", "Input della pipeline", "yaml"),
    Snippet("manifest_usb", "Manifest USB", "configs/delta_q_tracking/dataset_manifest.json", 5, 24, "Entry manifest per USB con PLY arricchito, RGB/mask e joint metadata.", "Input della pipeline", "json"),
    Snippet("manifest_storage", "Manifest Storage", "configs/delta_q_tracking/dataset_manifest.json", 174, 192, "Entry manifest Storage usata nel confronto finale.", "Input della pipeline", "json"),
    Snippet("rgb_mask_loading", "Caricamento RGB e mask", "scripts/delta_q_tracking/io_utils.py", 99, 115, "Carica frame RGB/mask e li converte in tensori CUDA.", "Input della pipeline", "python"),
    Snippet("camera_loading", "Costruzione camera COLMAP", "scripts/delta_q_tracking/io_utils.py", 210, 246, "Costruisce Camera per il renderer da intrinseci/estrinseci COLMAP.", "Input della pipeline", "python"),
    Snippet("load_freeze_gaussian", "Load e freeze Gaussian", "scripts/delta_q_tracking/io_utils.py", 142, 154, "Carica GaussianModel da PLY e disabilita i gradienti sui tensori 3DGS base.", "Caricamento e freeze", "python"),
    Snippet("ply_metadata", "Metadata articolata nel PLY", "scripts/delta_q_tracking/io_utils.py", 157, 196, "Legge joint_part, joint_type_id, joint_origin e joint_axis.", "Metadata articolata", "python"),
    Snippet("cli_motion_param", "Selezione direct_delta_q / mlp_q", "scripts/delta_q_tracking/run_sequence.py", 871, 921, "Definisce CLI e parametro --motion-param.", "Overview tecnico", "python"),
    Snippet("mode_setup", "Setup della parametrizzazione del moto", "scripts/delta_q_tracking/run_sequence.py", 1015, 1059, "Crea base_xyz, q_ref e, se richiesto, MotionMLP con optimizer Adam.", "Overview tecnico", "python"),
    Snippet("sequence_branch", "Branch per-frame Direct vs MLP", "scripts/delta_q_tracking/run_sequence.py", 1150, 1240, "Nel loop di sequenza chiama optimize_step_mlp oppure optimize_step.", "Loop condiviso", "python"),
    Snippet("articulation_transforms", "Trasformazioni prismatic/revolute", "scripts/delta_q_tracking/articulation.py", 174, 236, "Applica la cinematica esplicita ai soli Gaussian mobili.", "Deformazione cinematica", "python"),
    Snippet("rotation_update", "Aggiornamento quaternion", "scripts/delta_q_tracking/articulation.py", 239, 252, "Aggiorna le rotazioni dei Gaussian mobili in rotation_mode rigido.", "Deformazione cinematica", "python"),
    Snippet("deformed_gaussian", "Wrapper DeformedGaussian", "scripts/delta_q_tracking/deformed_gaussian.py", 4, 46, "Sostituisce xyz/rotation e inoltra le altre proprietà al Gaussian base.", "Wrapper DeformedGaussian", "python"),
    Snippet("losses", "Loss RGB mascherata e SSIM", "scripts/delta_q_tracking/losses.py", 8, 70, "Implementa L1 mascherata e SSIM opzionale.", "Loop condiviso", "python"),
    Snippet("direct_loop", "Loop direct_delta_q", "scripts/delta_q_tracking/run_sequence.py", 394, 461, "Ottimizza direttamente uno scalare delta_q con Adam.", "Direct optimization", "python"),
    Snippet("direct_commit", "Commit direct_delta_q", "scripts/delta_q_tracking/run_sequence.py", 505, 540, "Committa best/final delta_q e renderizza lo stato finale.", "Direct optimization", "python"),
    Snippet("motion_mlp", "Classe MotionMLP", "scripts/delta_q_tracking/motion_mlp.py", 6, 51, "MLP temporale che mappa t normalizzato in q(t).", "MLP optimization", "python"),
    Snippet("mlp_loop", "Loop mlp_q", "scripts/delta_q_tracking/run_sequence.py", 650, 708, "Predice q0/q1, deriva delta_q, renderizza e aggiorna i pesi MLP.", "MLP optimization", "python"),
    Snippet("mlp_regularization", "Regolarizzazioni MLP", "scripts/delta_q_tracking/run_sequence.py", 289, 314, "Calcola smoothness, acceleration e monotonic loss sulla griglia temporale.", "MLP optimization", "python"),
    Snippet("outputs", "Output diagnostici", "scripts/delta_q_tracking/run_sequence.py", 1241, 1461, "Salva render, overlay, PLY deformati, trajectory e log.", "Output e diagnostiche", "python"),
]


def extract(snippet):
    path = REPO_ROOT / snippet.file
    lines = path.read_text().splitlines()
    selected = lines[snippet.start_line - 1 : snippet.end_line]
    width = len(str(snippet.end_line))
    return "\n".join(
        "%*d | %s" % (width, line_number, line)
        for line_number, line in enumerate(selected, start=snippet.start_line)
    ) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for snippet in SNIPPETS:
        out_path = OUT_DIR / ("%s.txt" % snippet.slug)
        out_path.write_text(extract(snippet))
        index.append(snippet.to_json())
    (OUT_DIR / "snippets_index.json").write_text(json.dumps(index, indent=2))
    print("Estratti %d snippet in %s" % (len(SNIPPETS), OUT_DIR))


if __name__ == "__main__":
    main()
