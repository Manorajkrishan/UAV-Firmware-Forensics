import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Box,
  Paper,
  Typography,
  Alert,
  CircularProgress,
  Chip,
  Grid,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Divider,
  Button,
  Tabs,
  Tab,
} from '@mui/material'
import {
  ArrowBack as BackIcon,
  Timeline as TimelineIcon,
  Description as LogsIcon,
  VerifiedUser as EvidenceIcon,
  BarChart as VisualsIcon,
  Security as InjectionIcon,
} from '@mui/icons-material'
import { getAnalysis } from '../services/api'
import TamperingStatusGauge from '../components/visualizations/TamperingStatusGauge'
import BehaviorPieChart from '../components/visualizations/BehaviorPieChart'
import TimelineChart from '../components/visualizations/TimelineChart'
import FeatureContributionChart from '../components/visualizations/FeatureContributionChart'
import EvidencePanel from '../components/visualizations/EvidencePanel'
import InjectionDetectionPanel from '../components/visualizations/InjectionDetectionPanel'

function Analysis() {
  const { firmwareId } = useParams()
  const navigate = useNavigate()
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState(0)

  useEffect(() => {
    loadAnalysis()
  }, [firmwareId])

  const loadAnalysis = async () => {
    try {
      setLoading(true)
      const data = await getAnalysis(firmwareId)
      setAnalysis(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load analysis')
    } finally {
      setLoading(false)
    }
  }

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue)
  }

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Box>
        <Alert severity="error">{error}</Alert>
        <Button startIcon={<BackIcon />} onClick={() => navigate('/history')} sx={{ mt: 2 }}>
          Back to History
        </Button>
      </Box>
    )
  }

  if (!analysis) {
    return <Alert severity="warning">Analysis not found</Alert>
  }

  const tamperingProb = analysis.tampering_probability !== null && analysis.tampering_probability !== undefined 
    ? analysis.tampering_probability 
    : 0.5
  const confidence = analysis.confidence_score !== null && analysis.confidence_score !== undefined
    ? analysis.confidence_score
    : Math.abs(tamperingProb - 0.5) * 2 * 100
  const tamperingStatus = analysis.tampering_status || (analysis.is_tampered ? 'Tampered' : 'Untampered')
  const severityLevel = analysis.severity_level || 'Unknown'
  
  // Determine risk level
  let riskLevel = 'Low'
  let riskColor = 'success'
  if (severityLevel === 'Critical' || tamperingStatus === 'Tampered') {
    riskLevel = 'Critical'
    riskColor = 'error'
  } else if (severityLevel === 'High' || tamperingStatus === 'Suspicious') {
    riskLevel = 'High'
    riskColor = 'warning'
  } else if (severityLevel === 'Medium') {
    riskLevel = 'Medium'
    riskColor = 'warning'
  }

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={3}>
        <Button startIcon={<BackIcon />} onClick={() => navigate('/history')} sx={{ mr: 2 }}>
          Back
        </Button>
        <Typography variant="h4">Forensic Analysis Report</Typography>
      </Box>

      {/* Top Section: Case Info, Firmware Status, Risk Indicator */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Case Information
              </Typography>
              <Typography variant="h6" gutterBottom>
                {analysis.file_name || 'Unknown File'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                ID: {analysis.firmware_id}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Date: {analysis.timestamp ? new Date(analysis.timestamp).toLocaleString() : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Firmware Status
              </Typography>
              <Chip
                label={tamperingStatus}
                color={
                  tamperingStatus === 'Tampered' ? 'error' :
                  tamperingStatus === 'Suspicious' ? 'warning' :
                  'success'
                }
                sx={{ fontSize: '1.1rem', fontWeight: 'bold', py: 1, px: 2 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Model: {analysis.model_used || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Risk Indicator
              </Typography>
              <Chip
                label={severityLevel}
                color={riskColor}
                sx={{ fontSize: '1.1rem', fontWeight: 'bold', py: 1, px: 2 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Severity Score: {analysis.severity_score ? (analysis.severity_score * 100).toFixed(1) : 'N/A'}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tampering Status Gauge - Large, Center */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'center' }}>
        <Box sx={{ maxWidth: 500, width: '100%' }}>
          <TamperingStatusGauge
            probability={tamperingProb}
            status={tamperingStatus}
            confidence={confidence}
          />
        </Box>
      </Box>

      {/* Tabbed Interface */}
      <Paper elevation={3} sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab icon={<TimelineIcon />} iconPosition="start" label="Timeline" />
          <Tab icon={<LogsIcon />} iconPosition="start" label="Logs & Recommendations" />
          <Tab icon={<EvidenceIcon />} iconPosition="start" label="Evidence" />
          <Tab icon={<InjectionIcon />} iconPosition="start" label="Injection Detection" />
          <Tab icon={<VisualsIcon />} iconPosition="start" label="Visualizations" />
        </Tabs>

        {/* Tab Content */}
        <Box sx={{ p: 3 }}>
          {activeTab === 0 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Anomaly Timeline
              </Typography>
              <TimelineChart
                timelineData={analysis.timeline_data}
                anomalyThreshold={0.5}
              />
            </Box>
          )}

          {activeTab === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Analysis Logs & Recommendations
              </Typography>
              
              <Grid container spacing={3} sx={{ mt: 1 }}>
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Analysis Details
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText
                          primary="Tampering Probability"
                          secondary={`${(tamperingProb * 100).toFixed(2)}%`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Anomaly Score"
                          secondary={
                            analysis.anomaly_score !== null && analysis.anomaly_score !== undefined
                              ? (typeof analysis.anomaly_score === 'number' ? analysis.anomaly_score.toFixed(4) : String(analysis.anomaly_score))
                              : 'N/A'
                          }
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Confidence Score"
                          secondary={`${confidence.toFixed(1)}%`}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText
                          primary="Model Accuracy"
                          secondary={
                            analysis.model_accuracy !== null && analysis.model_accuracy !== undefined
                              ? `${(analysis.model_accuracy * 100).toFixed(1)}%`
                              : 'N/A'
                          }
                        />
                      </ListItem>
                      {analysis.tampering_type && (
                        <ListItem>
                          <ListItemText
                            primary="Tampering Type"
                            secondary={analysis.tampering_type}
                          />
                        </ListItem>
                      )}
                      {analysis.time_window && (
                        <ListItem>
                          <ListItemText
                            primary="Time Window"
                            secondary={analysis.time_window}
                          />
                        </ListItem>
                      )}
                    </List>
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Recommendations
                    </Typography>
                    {analysis.recommendations && Array.isArray(analysis.recommendations) && analysis.recommendations.length > 0 ? (
                      <List dense>
                        {analysis.recommendations
                          .filter(rec => rec !== null && rec !== undefined)
                          .map((rec, index) => (
                            <ListItem key={index}>
                              <ListItemText primary={`• ${String(rec)}`} />
                            </ListItem>
                          ))}
                      </List>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        No recommendations available
                      </Typography>
                    )}
                  </Paper>
                </Grid>
              </Grid>
            </Box>
          )}

          {activeTab === 2 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Evidence & Integrity
              </Typography>
              <EvidencePanel analysis={analysis} />
            </Box>
          )}

          {activeTab === 3 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Injection Detection Analysis
              </Typography>
              <InjectionDetectionPanel analysis={analysis} />
            </Box>
          )}

          {activeTab === 4 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Visualizations
              </Typography>
              <Grid container spacing={3} sx={{ mt: 1 }}>
                <Grid item xs={12} md={6}>
                  <BehaviorPieChart tamperingProbability={tamperingProb} />
                </Grid>
                <Grid item xs={12} md={6}>
                  <FeatureContributionChart
                    featureContributions={analysis.feature_contributions}
                  />
                </Grid>
              </Grid>
            </Box>
          )}
        </Box>
      </Paper>
    </Box>
  )
}

export default Analysis
