# SAM-ARTGS Code Report

Report tecnico HTML in italiano sulla pipeline:

```text
gaussian-splatting/scripts/delta_q_tracking/
```

Messaggio principale:

```text
Same pipeline, different motion parameterization.
```

## Aprire il report

Metodo consigliato:

```bash
cd /leonardo_work/IscrC_EditGS/andrea/samartgs/gaussian-splatting
python3 -m http.server 8000
```

Poi aprire:

```text
http://localhost:8000/docs/samartgs_code_report/
```

Servire dalla root `gaussian-splatting/` permette sia di caricare gli snippet del report sia di aprire i link reali verso `outputs/delta_q_tracking/`.

## Syntax highlighting

Il report usa Pygments per generare snippet evidenziati su sfondo chiaro.
Se Pygments non è disponibile:

```bash
pip install pygments
```

Per rigenerare gli snippet HTML evidenziati:

```bash
cd /leonardo_work/IscrC_EditGS/andrea/samartgs/gaussian-splatting/docs/samartgs_code_report
python highlight_snippets.py
cd ../..
python -m http.server 8000
```

Lo script legge `assets/snippets/*.txt` e scrive:

```text
assets/highlighted_snippets/*.html
assets/highlighted_snippets/pygments.css
assets/highlighted_snippets/highlighted_index.json
```

## Aggiornare gli snippet

Da `gaussian-splatting/`:

```bash
python3 docs/samartgs_code_report/extract_code_snippets.py
cd docs/samartgs_code_report
python3 highlight_snippets.py
```

Lo script legge i file reali della repo ed esporta:

```text
docs/samartgs_code_report/assets/snippets/*.txt
docs/samartgs_code_report/assets/snippets/snippets_index.json
```

Ogni entry JSON contiene:

- `title`
- `file`
- `start_line`
- `end_line`
- `purpose`
- `section`

## Dove mettere screenshot, plot e video

Directory previste:

```text
docs/samartgs_code_report/assets/results/
docs/samartgs_code_report/assets/screenshots/
docs/samartgs_code_report/assets/videos/
```

La sezione risultati del report principale non usa placeholder: punta direttamente agli output reali in:

```text
outputs/delta_q_tracking/final_direct_vs_mlp/
outputs/delta_q_tracking/new_dataset/
```

La cartella `assets/results/` deve contenere solo eventuali copie di output reali prodotti dalla pipeline o dagli script di reporting.
Non inserire placeholder grafici finti nel report.

Asset reali utili, se generati:

- `assets/results/q_ref_vs_gt.png`
- `assets/results/delta_q_vs_gt.png`
- `assets/results/direct_overlay.mp4`
- `assets/results/mlp_overlay.mp4`
- `assets/results/comparison_table.png`

## Asset mancanti

- Nessun placeholder richiesto.
- Copiare asset in `assets/results/` solo se serve rendere il report autosufficiente fuori dalla repo.
- Se un output non esiste in `outputs/delta_q_tracking/`, non mostrarlo nel report.

## Sezioni da verificare

- Uso effettivo di `use_depth` e `depth_dir`.
- Uso effettivo dei parametri di alignment nel loop principale.
- Utility potenzialmente legacy da separare dalla pipeline attuale.
- Oggetti nel manifest con `gaussian_model_path: null`.
- Generalità della MLP: il codice attuale la usa come ottimizzazione sequence-specific.

## File sorgente coperti

- `scripts/delta_q_tracking/run_sequence.py`
- `scripts/delta_q_tracking/config_usb.yaml`
- `scripts/delta_q_tracking/motion_mlp.py`
- `scripts/delta_q_tracking/io_utils.py`
- `scripts/delta_q_tracking/articulation.py`
- `scripts/delta_q_tracking/deformed_gaussian.py`
- `scripts/delta_q_tracking/losses.py`
- `configs/delta_q_tracking/dataset_manifest.json`
