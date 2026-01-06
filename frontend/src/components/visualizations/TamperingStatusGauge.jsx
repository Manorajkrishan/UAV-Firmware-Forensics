import React from 'react'
import { Box, Typography, CircularProgress, Paper } from '@mui/material'

function TamperingStatusGauge({ probability, status, confidence }) {
  const percentage = typeof probability === 'number' ? probability * 100 : 0
  const confidenceScore = typeof confidence === 'number' ? confidence : 0
  
  // Determine color based on status
  let color = '#4caf50' // Green for Untampered
  if (status === 'Tampered') {
    color = '#f44336' // Red
  } else if (status === 'Suspicious') {
    color = '#ff9800' // Orange
  }
  
  // Determine severity text
  let statusText = status || 'Unknown'
  let statusColor = color
  
  return (
    <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
      <Typography variant="h6" gutterBottom>
        Tampering Status
      </Typography>
      
      <Box sx={{ position: 'relative', display: 'inline-flex', mb: 2 }}>
        <CircularProgress
          variant="determinate"
          value={percentage}
          size={200}
          thickness={8}
          sx={{
            color: color,
            '& .MuiCircularProgress-circle': {
              strokeLinecap: 'round',
            },
          }}
        />
        <Box
          sx={{
            top: 0,
            left: 0,
            bottom: 0,
            right: 0,
            position: 'absolute',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
          }}
        >
          <Typography variant="h4" component="div" sx={{ color: statusColor, fontWeight: 'bold' }}>
            {percentage.toFixed(1)}%
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
            {statusText}
          </Typography>
        </Box>
      </Box>
      
      <Box sx={{ mt: 2 }}>
        <Typography variant="body1" color="text.secondary">
          Confidence: {confidenceScore.toFixed(1)}%
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Tampering Probability
        </Typography>
      </Box>
    </Paper>
  )
}

export default TamperingStatusGauge

