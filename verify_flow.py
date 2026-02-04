import httpx
import asyncio
import json
import os
import sys

async def check():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. Start a trace via Chat
        print("Starting chat...")
        chat_req = {'message': 'Trace 0x742At94420A80C5556209f83C7E3A06CdAA0F425 on eth', 'user_id': 'test_user'}
        resp = await client.post('http://localhost:8000/api/chat', json=chat_req)
        print(f"Chat Start Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            return
            
        session_id = resp.json().get('session_id')
        print(f"Session ID: {session_id}")
        
        # 2. Confirm to start trace
        print("Sending confirmation...")
        confirm_req = {'message': 'yes', 'session_id': session_id, 'user_id': 'test_user'}
        result_received = False
        async with client.stream('POST', 'http://localhost:8000/api/chat', json=confirm_req) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'status':
                            print(f"Status: {data.get('status')}")
                        if data.get('type') == 'result':
                            result_received = True
                            report = data.get('report')
                            print('--- Chat Report Received ---')
                            summary = report.get('summary_text', '')
                            print(f"Summary (start): {summary[:50]}...")
                            has_ascii = 'ascii_tree' in report and report.get('ascii_tree')
                            print(f"ASCII Tree Present: {bool(has_ascii)}")
                            print(f"Report Keys: {list(report.keys())}")
                            
                            # Verify full data is NOT in report
                            full_keys_in_report = [k for k in ['paths', 'entities', 'annotations'] if k in report]
                            if full_keys_in_report:
                                print(f"ERROR: Report contains full trace data keys: {full_keys_in_report}")
                            else:
                                print("SUCCESS: Report is streamlined for chat.")
                    except Exception as e:
                        # print(f"Error parsing line: {e}")
                        pass
        
        if not result_received:
            print("ERROR: Result not received from chat streaming.")
            return

        # 3. Call /api/visualization
        print("Calling /api/visualization...")
        viz_req = {'session_id': session_id}
        resp = await client.post('http://localhost:8000/api/visualization', json=viz_req, timeout=30.0)
        print(f"Visualization Endpoint Status: {resp.status_code}")
        if resp.status_code == 200:
            viz_data = resp.json()
            print(f"Viz Data Keys: {list(viz_data.keys())}")
            
            # 4. Call /api/visualization/share
            print("Calling /api/visualization/share...")
            share_req = {'visualization_data': viz_data}
            resp = await client.post('http://localhost:8000/api/visualization/share', json=share_req, timeout=30.0)
            print(f"Share Endpoint Status: {resp.status_code}")
            if resp.status_code == 200:
                 print(f"Share URL: {resp.json().get('share_url')}")
        else:
             print(f"Error: {resp.text}")

if __name__ == '__main__':
    asyncio.run(check())
