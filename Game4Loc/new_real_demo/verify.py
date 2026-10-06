"""Validate the real output against source frames, metadata and satellite crops."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from build import ROOT, OUT, SOURCE, Satellite, PW, PH, LX, RX, TOP
from telemetry import read_telemetry


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    rows=list(csv.DictReader((OUT/'frames.csv').open()))
    info,times,packets=read_telemetry(SOURCE)
    start=manifest['clip_start_source_frame']
    n=manifest['frame_count']
    assert len(rows)==n
    assert [int(r['frame_index']) for r in rows]==list(range(start,start+n))
    for i,row in enumerate(rows):
        p=packets[start+i]
        assert abs(float(row['video_time_s'])-times[start+i])<1e-9
        assert abs(float(row['raw_lat'])-p['lat'])<1e-10
        assert abs(float(row['raw_lon'])-p['lon'])<1e-10
        assert abs(float(row['telemetry_delta_s']))<0.001
    for name in ['new_real_demo.mp4','satellite_frames.mp4']:
        cap=cv2.VideoCapture(str(OUT/name))
        assert abs(cap.get(cv2.CAP_PROP_FPS)-info['fps'])<1e-5
        count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame is not None
            count+=1
        cap.release()
        assert count==n,(name,count,n)
    sat=Satellite(Path(manifest['satellite_cache']),manifest['base_zoom'],Path(manifest['satellite_detail_cache']) if manifest.get('satellite_detail_cache') else None)
    source=cv2.VideoCapture(str(SOURCE))
    demo=cv2.VideoCapture(str(OUT/'new_real_demo.mp4'))
    satellite_video=cv2.VideoCapture(str(OUT/'satellite_frames.mp4'))
    checks=[]
    for i in np.linspace(0,n-1,7).astype(int):
        source.set(cv2.CAP_PROP_POS_FRAMES,start+i);ok,raw=source.read();assert ok
        demo.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,shown=demo.read();assert ok
        satellite_video.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,satframe=satellite_video.read();assert ok
        expected=cv2.resize(raw,(PW,PH),interpolation=cv2.INTER_AREA)
        right=shown[TOP:TOP+PH,RX:RX+PW]
        source_mae=float(np.abs(expected.astype(float)-right).mean())
        row={k:float(rows[i][k]) for k in ['lat','lon','height_m','yaw_deg','pitch_deg','roll_deg']}
        patch,*_=sat.render(row,manifest['horizontal_fov_deg'])
        left=shown[TOP:TOP+PH,LX:LX+PW]
        satellite_mae=float(np.abs(patch.astype(float)-left).mean())
        satellite_video_mae=float(np.abs(patch.astype(float)-satframe[:PH,:PW]).mean())
        assert source_mae<6,(i,'wrong source frame',source_mae)
        assert satellite_mae<5,(i,'wrong satellite crop',satellite_mae)
        assert satellite_video_mae<5,(i,'wrong satellite-only video',satellite_video_mae)
        checks.append(dict(output_frame=int(i),source_frame=int(start+i),
                           source_mae=source_mae,satellite_mae=satellite_mae,
                           satellite_only_mae=satellite_video_mae))
    source.release();demo.release();satellite_video.release()
    report=json.loads((OUT/'verification.json').read_text())
    report.update(full_decode_frame_count=n,frame_mapping_rows_checked=len(rows),
                  visual_sample_checks=checks,status='PASS')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
