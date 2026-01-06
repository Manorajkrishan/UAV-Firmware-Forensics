import React from 'react'
import { Paper, Typography, Box } from '@mui/material'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

function BehaviorPieChart({ tamperingProbability }) {
  const normalPercent = typeof tamperingProbability === 'number' 
    ? (1 - tamperingProbability) * 100 
    : 100
  const anomalousPercent = typeof tamperingProbability === 'number' 
    ? tamperingProbability * 100 
    : 0
  
  const data = [
    { name: 'Normal Behavior', value: normalPercent, color: '#4caf50' },
    { name: 'Anomalous Behavior', value: anomalousPercent, color: '#f44336' }
  ]
  
  const COLORS = {
    'Normal Behavior': '#4caf50',
    'Anomalous Behavior': '#f44336'
  }
  
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Behavior Distribution
      </Typography>
      <Box sx={{ width: '100%', height: 300, mt: 2 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[entry.name]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  )
}

export default BehaviorPieChart

