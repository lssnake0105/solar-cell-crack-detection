import os
import time
import zipfile
import argparse

ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), 'archives')

INCLUDE_PATHS = [
    # Core outputs
    os.path.join('runs', 'unet', 'best.pt'),
    os.path.join('runs', 'unet', 'last.pt'),
    os.path.join('runs', 'mask_stats.csv'),
    # Visual validation
    os.path.join('overlays', 'val'),
    # Docs & scripts
    'README.md',
    'requirements_seg.txt',
    'train_unet.py',
    'infer_overlay.py',
    'infer_dir.py',
    'check_masks.py',
]


def add_file(zf: zipfile.ZipFile, root: str, rel_path: str):
    full_path = os.path.join(root, rel_path)
    if os.path.isfile(full_path):
        zf.write(full_path, arcname=rel_path)
    elif os.path.isdir(full_path):
        # add dir contents
        for dirpath, _, filenames in os.walk(full_path):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                rp = os.path.relpath(fp, root)
                zf.write(fp, arcname=rp)


def main():
    parser = argparse.ArgumentParser(description='Archive current UNet training outputs for acceptance')
    parser.add_argument('--root', default=os.path.dirname(__file__), help='svd_seg root')
    args = parser.parse_args()

    root = args.root
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_zip = os.path.join(ARCHIVE_DIR, f'unet_run_{ts}.zip')

    # Optional summary
    summary_txt = os.path.join(root, 'runs', 'SUMMARY.txt')
    try:
        best_pt = os.path.join(root, 'runs', 'unet', 'best.pt')
        last_pt = os.path.join(root, 'runs', 'unet', 'last.pt')
        overlays_val = os.path.join(root, 'overlays', 'val')
        lines = []
        lines.append('UNet Training Archive Summary')
        lines.append(f'Timestamp: {ts}')
        lines.append(f'best.pt: {os.path.getsize(best_pt)} bytes' if os.path.exists(best_pt) else 'best.pt: MISSING')
        lines.append(f'last.pt: {os.path.getsize(last_pt)} bytes' if os.path.exists(last_pt) else 'last.pt: MISSING')
        if os.path.isdir(overlays_val):
            cnt = len([f for f in os.listdir(overlays_val) if f.lower().endswith('.png')])
            lines.append(f'val overlays: {cnt} files')
        else:
            lines.append('val overlays: MISSING')
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    except Exception as e:
        print(f'[WARN] failed to build summary: {e}')
        summary_txt = None

    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_PATHS:
            add_file(zf, root, rel)
        # add summary if built
        if summary_txt and os.path.exists(summary_txt):
            zf.write(summary_txt, arcname=os.path.join('runs', 'SUMMARY.txt'))

    # clean summary temp
    try:
        if summary_txt and os.path.exists(summary_txt):
            os.remove(summary_txt)
    except:
        pass

    print(f'[DONE] Archive created: {out_zip}')


if __name__ == '__main__':
    main()
