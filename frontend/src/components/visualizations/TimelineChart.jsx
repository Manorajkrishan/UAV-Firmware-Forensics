import React from 'react'
import { Paper, Typography, Box } from '@mui/material'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts'

function TimelineChart({ timelineData, anomalyThreshold = 0.5 }) {
  // If no timeline data, create sample data
  const data = timelineData && timelineData.length > 0 
    ? timelineData 
    : Array.from({ length: 20 }, (_, i) => ({
        time: `Sample ${i + 1}`,
        anomaly_score: Math.random() * 0.8,
        tampering_probability: Math.random() * 0.7,
        is_anomaly: Math.random() > 0.7
      }))
  
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Anomaly Timeline
      </Typography>
      <Box sx={{ width: '100%', height: 400, mt: 2 }}>
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="time" 
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis 
              domain={[0, 1]}
              label={{ value: 'Anomaly Score', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip />
            <Legend />
            <ReferenceLine 
              y={anomalyThreshold} 
              stroke="#ff9800" 
              strokeDasharray="5 5"
              label="Threshold"
            />
            <Line 
              type="monotone" 
              dataKey="anomaly_score" 
              stroke="#f44336" 
              strokeWidth={2}
              name="Anomaly Score"
              dot={{ r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="tampering_probability" 
              stroke="#2196f3" 
              strokeWidth={2}
              name="Tampering Probability"
              strokeDasharray="5 5"
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  )
}

export default TimelineChart

