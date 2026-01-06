import React from 'react'
import {
  Paper,
  Typography,
  Box,
  Chip,
  Grid,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Divider,
  Alert
} from '@mui/material'
import {
  CheckCircle,
  Warning,
  Error,
  Security,
  Code,
  Settings,
  BugReport,
  Sensors
} from '@mui/icons-material'

function InjectionDetectionPanel({ analysis }) {
  const injectionAnalysis = analysis?.injection_analysis || {}
  const mitreClassification = analysis?.mitre_classification
  
  const injectionDetected = injectionAnalysis?.injection_detected || false
  const injectionStatus = injectionAnalysis?.injection_status || 'Unknown'
  const injectionType = injectionAnalysis?.injection_type || 'Unknown'
  const injectionTypes = injectionAnalysis?.injection_types || []
  const confidence = injectionAnalysis?.confidence || 0
  const evidence = injectionAnalysis?.evidence || []
  const activationTimeline = injectionAnalysis?.activation_timeline
  
  // Execution flow anomalies
  const executionAnomalies = injectionAnalysis?.execution_flow_anomalies || {}
  const sensorSpoofing = injectionAnalysis?.sensor_spoofing || {}
  const safetyBypass = injectionAnalysis?.safety_system_bypass || {}
  
  // Determine status color
  let statusColor = 'success'
  let statusIcon = <CheckCircle />
  if (injectionStatus === 'Injected') {
    statusColor = 'error'
    statusIcon = <Error />
  } else if (injectionStatus === 'Suspicious') {
    statusColor = 'warning'
    statusIcon = <Warning />
  }
  
  // Injection type icons
  const injectionTypeIcons = {
    'Code Injection': <Code />,
    'Bootloader Injection': <Security />,
    'Configuration Injection': <Settings />,
    'Backdoor Injection': <BugReport />,
    'Sensor Spoofing': <Sensors />,
    'General Injection': <Error />
  }
  
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Injection Detection Analysis
      </Typography>
      
      {/* MITRE Classification */}
      {mitreClassification && (
        <Alert severity={mitreClassification.toLowerCase() === 'untampered' ? 'success' : 'warning'} sx={{ mb: 2 }}>
          <Typography variant="subtitle2">MITRE Classification: {mitreClassification}</Typography>
        </Alert>
      )}
      
      {/* Injection Status Card */}
      <Card sx={{ mb: 3, bgcolor: injectionDetected ? 'error.light' : 'success.light' }}>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2}>
            {statusIcon}
            <Box>
              <Typography variant="h6">
                Injection Status: {injectionStatus}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Confidence: {(confidence * 100).toFixed(1)}%
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
      
      {/* Injection Type */}
      {injectionType && injectionType !== 'Unknown' && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Detected Injection Type
          </Typography>
          <Chip
            icon={injectionTypeIcons[injectionType] || <Error />}
            label={injectionType}
            color={injectionDetected ? 'error' : 'default'}
            sx={{ fontSize: '1rem', py: 2, px: 1 }}
          />
          {injectionTypes.length > 1 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Additional Types Detected:
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {injectionTypes.filter(t => t !== injectionType).map((type, idx) => (
                  <Chip key={idx} label={type} size="small" variant="outlined" />
                ))}
              </Box>
            </Box>
          )}
        </Box>
      )}
      
      <Divider sx={{ my: 2 }} />
      
      {/* Detailed Analysis */}
      <Grid container spacing={2}>
        {/* Execution Flow Anomalies */}
        {executionAnomalies && Object.keys(executionAnomalies).length > 0 && (
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  Execution Flow Anomalies
                </Typography>
                {executionAnomalies.unexpected_calls && executionAnomalies.unexpected_calls.length > 0 && (
                  <List dense>
                    {executionAnomalies.unexpected_calls.map((call, idx) => (
                      <ListItem key={idx}>
                        <ListItemText primary={call} />
                      </ListItem>
                    ))}
                  </List>
                )}
                {executionAnomalies.rare_sequences && executionAnomalies.rare_sequences.length > 0 && (
                  <List dense>
                    {executionAnomalies.rare_sequences.map((seq, idx) => (
                      <ListItem key={idx}>
                        <ListItemText primary={seq} />
                      </ListItem>
                    ))}
                  </List>
                )}
                {executionAnomalies.unauthorized_routines && executionAnomalies.unauthorized_routines.length > 0 && (
                  <List dense>
                    {executionAnomalies.unauthorized_routines.map((routine, idx) => (
                      <ListItem key={idx}>
                        <ListItemText primary={routine} />
                      </ListItem>
                    ))}
                  </List>
                )}
                {executionAnomalies.anomaly_score !== undefined && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Anomaly Score: {(executionAnomalies.anomaly_score * 100).toFixed(1)}%
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}
        
        {/* Sensor Spoofing */}
        {sensorSpoofing && (sensorSpoofing.gps_spoofing || sensorSpoofing.altitude_spoofing || sensorSpoofing.battery_spoofing) && (
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  Sensor Spoofing Detection
                </Typography>
                <List dense>
                  {sensorSpoofing.gps_spoofing && (
                    <ListItem>
                      <ListItemText 
                        primary="GPS Spoofing Detected"
                        secondary="Fake GPS coordinates detected"
                      />
                    </ListItem>
                  )}
                  {sensorSpoofing.altitude_spoofing && (
                    <ListItem>
                      <ListItemText 
                        primary="Altitude Spoofing Detected"
                        secondary="False altitude readings detected"
                      />
                    </ListItem>
                  )}
                  {sensorSpoofing.battery_spoofing && (
                    <ListItem>
                      <ListItemText 
                        primary="Battery Spoofing Detected"
                        secondary="Fake battery status detected"
                      />
                    </ListItem>
                  )}
                </List>
                {sensorSpoofing.evidence && sensorSpoofing.evidence.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    {sensorSpoofing.evidence.map((ev, idx) => (
                      <Typography key={idx} variant="caption" display="block" color="text.secondary">
                        • {ev}
                      </Typography>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}
        
        {/* Safety System Bypass */}
        {safetyBypass && (safetyBypass.failsafe_disabled || safetyBypass.geofencing_disabled || safetyBypass.emergency_bypass) && (
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ borderColor: 'error.main' }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom color="error">
                  Safety System Bypass
                </Typography>
                <List dense>
                  {safetyBypass.failsafe_disabled && (
                    <ListItem>
                      <ListItemText 
                        primary="Failsafe Disabled"
                        secondary="Critical safety mechanism bypassed"
                      />
                    </ListItem>
                  )}
                  {safetyBypass.geofencing_disabled && (
                    <ListItem>
                      <ListItemText 
                        primary="Geofencing Disabled"
                        secondary="Geographic restrictions removed"
                      />
                    </ListItem>
                  )}
                  {safetyBypass.emergency_bypass && (
                    <ListItem>
                      <ListItemText 
                        primary="Emergency Bypass"
                        secondary="Emergency protocols bypassed"
                      />
                    </ListItem>
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
      
      {/* Evidence */}
      {evidence && evidence.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Evidence
          </Typography>
          <List dense>
            {evidence.map((ev, idx) => (
              <ListItem key={idx}>
                <ListItemText primary={`• ${ev}`} />
              </ListItem>
            ))}
          </List>
        </Box>
      )}
      
      {/* Activation Timeline */}
      {activationTimeline && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Activation Timeline
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {activationTimeline}
          </Typography>
        </Box>
      )}
    </Paper>
  )
}

export default InjectionDetectionPanel

