export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const E2E_API_KEY = process.env.E2E_API_KEY || 'd0fd4abe-a0db-4ac7-9302-978095f2a850';
  const E2E_AUTH_TOKEN = process.env.E2E_AUTH_TOKEN || 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJGSjg2R2NGM2pUYk5MT2NvNE52WmtVQ0lVbWZZQ3FvcXRPUWVNZmJoTmxFIn0.eyJleHAiOjE4MjIwMjU4MDQsImlhdCI6MTc5MDQ4OTgwNCwianRpIjoiYjI0YjZmMTItM2I4Yy00NzdlLWI2NjQtODBhYzRmMjllZDVhIiwiaXNzIjoiaHR0cDovL2dhdGV3YXkuZTJlbmV0d29ya3MuY29tL2F1dGgvcmVhbG1zL2FwaW1hbiIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiIzYTdiMTRjYy03YzAzLTQ4YTYtOGVjZC0zZTBhMTM5NGRjOGEiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGltYW51aSIsInNlc3Npb25fc3RhdGUiOiI1OTAxYzc1NS00NThlLTQ4MzYtOTM2Ny1hOTg0MmZhZWMzYmQiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImFwaXVzZXIiLCJkZWZhdWx0LXJvbGVzLWFwaW1hbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6IjU5MDFjNzU1LTQ1OGUtNDgzNi05MzY3LWE5ODQyZmFlYzNiZCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNfcGFydG5lcl9yb2xlIjpmYWxzZSwibmFtZSI6IlNhcnZlc2ggTWFoYWphbiIsInByaW1hcnlfZW1haWwiOiJzYXJ2ZXNobTQ0NDRAZ21haWwuY29tIiwiaXNfcHJpbWFyeV9jb250YWN0Ijp0cnVlLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJzYXJ2ZXNobTQ0NDRAZ21haWwuY29tIiwiZ2l2ZW5fbmFtZSI6IlNhcnZlc2giLCJmYW1pbHlfbmFtZSI6Ik1haGFqYW4iLCJlbWFpbCI6InNhcnZlc2htNDQ0NEBnbWFpbC5jb20iLCJpc19pbmRpYWFpX3VzZXIiOmZhbHNlfQ.fOCEDyM703_-hpBrg9tfBkBZ_eW37YPgyiM9h9uPMpt6PQpHuZRnYp0YQom2tGDlvam274Y3NUoemInVZzxfofuqEZxy-gbEpoEuMskbf953y_qiw9XZ4zM9IhXR-aK1o5OYv0reivhKcjImrmN2W1Yx8MgAGRIza0XpaF-HYoc';
  const NODE_ID = '347558';
  const LOCATION = 'Chennai';

  const action = req.query.action || (req.body && req.body.action) || 'status';

  // 1. Check Node Status
  if (action === 'status') {
    try {
      const url = `https://api.e2enetworks.com/myaccount/api/v1/nodes/?apikey=${E2E_API_KEY}&location=${LOCATION}`;
      const r = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${E2E_AUTH_TOKEN}`,
          'Accept': 'application/json'
        }
      });
      const data = await r.json();
      const node = (data.data || []).find(n => String(n.id) === NODE_ID) || data.data?.[0];
      return res.status(200).json({
        node_id: NODE_ID,
        name: node?.name || 'C3-16GB-578',
        status: node?.status || 'Unknown',
        public_ip: node?.public_ip_address || '151.185.58.96'
      });
    } catch (err) {
      return res.status(500).json({ error: String(err) });
    }
  }

  // 2. Wake / Power On Node
  if (action === 'wake' || action === 'power_on') {
    try {
      const url = `https://api.e2enetworks.com/myaccount/api/v1/nodes/${NODE_ID}/actions/?apikey=${E2E_API_KEY}&location=${LOCATION}`;
      const r = await fetch(url, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${E2E_AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ type: 'power_on' })
      });
      const data = await r.json();
      const isAlreadyRunning = data.errors && String(data.errors).includes('already in state');
      return res.status(200).json({
        success: true,
        message: isAlreadyRunning ? 'Node is already running' : 'Power-on command sent successfully',
        status: 'Starting',
        e2e_response: data
      });
    } catch (err) {
      return res.status(500).json({ error: String(err) });
    }
  }

  // 3. Sleep / Power Off Node (Scale-to-Zero)
  if (action === 'sleep' || action === 'power_off') {
    try {
      const url = `https://api.e2enetworks.com/myaccount/api/v1/nodes/${NODE_ID}/actions/?apikey=${E2E_API_KEY}&location=${LOCATION}`;
      const r = await fetch(url, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${E2E_AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ type: 'power_off' })
      });
      const data = await r.json();
      return res.status(200).json({
        success: true,
        message: 'Power-off command sent successfully',
        status: 'Shutting down',
        e2e_response: data
      });
    } catch (err) {
      return res.status(500).json({ error: String(err) });
    }
  }

  return res.status(400).json({ error: `Unknown action: ${action}` });
}
