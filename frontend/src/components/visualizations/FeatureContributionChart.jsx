import React from 'react'
import { Paper, Typography, Box } from '@mui/material'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'

function FeatureContributionChart({ featureContributions }) {
  // Transform feature contributions data for chart
  const data = featureContributions 
    ? Object.entries(featureContributions)
        .map(([feature, data]) => ({
          feature: feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          contribution: data.percentage || 0,
          value: data.value || 0
        }))
        .sort((a, b) => b.contribution - a.contribution)
        .slice(0, 10) // Top 10 features
    : []
  
  if (data.length === 0) {
    return (
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Feature Contribution Analysis
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          No feature contribution data available
        </Typography>
      </Paper>
    )
  }
  
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Feature Contribution Analysis
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Top features contributing to tampering detection
      </Typography>
      <Box sx={{ width: '100%', height: 400, mt: 2 }}>
        <ResponsiveContainer>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} label={{ value: 'Contribution (%)', position: 'insideBottom', offset: -5 }} />
            <YAxis dataKey="feature" type="category" width={90} />
            <Tooltip 
              formatter={(value) => [`${value.toFixed(1)}%`, 'Contribution']}
            />
            <Legend />
            <Bar dataKey="contribution" fill="#2196f3" name="Contribution %" />
          </BarChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  )
}

export default FeatureContributionChart

