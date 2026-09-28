import urllib.request
import json
import os

def test_buildings():
    boundary = '----SatQueryBoundary12345'
    img_path = 'frontend/public/demo_buildings.tif'
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return

    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="query"\r\n\r\n'
        f"Detect and extract building footprints and count structures\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image_a"; filename="demo_buildings.tif"\r\n'
        f"Content-Type: image/tiff\r\n\r\n"
    ).encode('utf-8') + img_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')

    req = urllib.request.Request(
        'https://satquery-backend-620388855751.asia-south1.run.app/api/v1/analyze',
        data=body,
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'User-Agent': 'SatQuery-Client'
        }
    )
    print("Testing building inference on demo_buildings.tif...")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("SUCCESS!")
        print("Headline:", data.get('summary', {}).get('headline'))
        print("Metrics:", data.get('summary', {}).get('metrics'))
        for viz in data.get('visualizations', []):
            print(f"  - Layer: {viz.get('title')} ({viz.get('category')})")

if __name__ == '__main__':
    test_buildings()
