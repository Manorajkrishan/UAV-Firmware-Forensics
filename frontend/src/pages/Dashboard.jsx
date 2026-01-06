import React, { useEffect, useState } from 'react'
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Chip,
  Button,
} from '@mui/material'
import {
  Security as SecurityIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell } from 'recharts'
import { getDashboardStats, healthCheck } from '../services/api'
import { useNavigate } from 'react-router-dom'

const COLORS = ['#4caf50', '#f44336']

function Dashboard() {
  const [stats, setStats] = useState(null)
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    loadData()
    
    // Refresh data when component comes into focus (user navigates back)
    const handleFocus = () => {
      loadData()
    }
    window.addEventListener('focus', handleFocus)
    
    // Also refresh every 30 seconds
    const interval = setInterval(() => {
      loadData()
    }, 30000)
    
    return () => {
      window.removeEventListener('focus', handleFocus)
      clearInterval(interval)
    }
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Load data with individual error handling
      let statsData = {
        total_analyses: 0,
        tampered_count: 0,
        clean_count: 0,
        recent_analyses: []
      }
      let healthData = null
      
      try {
        statsData = await getDashboardStats()
      } catch (err) {
        console.warn('Stats load failed, using defaults:', err)
        // statsData already has defaults
      }
      
      try {
        healthData = await healthCheck()
      } catch (err) {
        console.warn('Health check failed:', err)
        // healthData remains null, component handles it
      }
      
      console.log('Dashboard stats loaded:', JSON.stringify(statsData, null, 2))
      console.log('Stats breakdown:', {
        total: statsData.total_analyses,
        clean: statsData.clean_count,
        tampered: statsData.tampered_count,
        recentCount: statsData.recent_analyses?.length || 0
      })
      setStats(statsData)
      setHealth(healthData)
    } catch (err) {
      console.error('Unexpected error loading dashboard:', err)
      // Don't set error state, just use defaults
      setStats({
        total_analyses: 0,
        tampered_count: 0,
        clean_count: 0,
        recent_analyses: []
      })
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return <Alert severity="error">Error loading dashboard: {error}</Alert>
  }

  const pieData = [
    { name: 'Clean', value: stats?.clean_count || 0 },
    { name: 'Tampered', value: stats?.tampered_count || 0 },
  ]

  const recentData = (stats?.recent_analyses || []).slice(0, 5).map((a) => ({
    name: a.file_name?.substring(0, 20) || 'Unknown',
    probability: (a.probability || a.tampering_probability || 0) * 100,
  }))

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h4">
          Dashboard
        </Typography>
        <Button 
          variant="outlined" 
          onClick={loadData}
          disabled={loading}
        >
          {loading ? <CircularProgress size={20} /> : 'Refresh'}
        </Button>
      </Box>

      {health && (
        <Alert
          severity={health.status === 'healthy' ? 'success' : 'warning'}
          sx={{ mb: 3 }}
        >
          System Status: {health.status.toUpperCase()} | Models Loaded: {health.models_loaded ? 'Yes' : 'No'}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Statistics Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <AssessmentIcon sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Total Analyses
                  </Typography>
                  <Typography variant="h4">{stats?.total_analyses || 0}</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <CheckIcon sx={{ fontSize: 40, color: 'success.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Clean Firmware
                  </Typography>
                  <Typography variant="h4">{stats?.clean_count || 0}</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <WarningIcon sx={{ fontSize: 40, color: 'error.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Tampered Firmware
                  </Typography>
                  <Typography variant="h4">{stats?.tampered_count || 0}</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <SecurityIcon sx={{ fontSize: 40, color: 'warning.main', mr: 2 }} />
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Threat Rate
                  </Typography>
                  <Typography variant="h4">
                    {stats && stats.total_analyses > 0
                      ? ((stats.tampered_count / stats.total_analyses) * 100).toFixed(1)
                      : 0}
                    %
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Firmware Distribution
            </Typography>
            <PieChart width={400} height={300}>
              <Pie
                data={pieData}
                cx={200}
                cy={150}
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Analysis Results
            </Typography>
            <BarChart width={500} height={300} data={recentData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="probability" fill="#8884d8" />
            </BarChart>
          </Paper>
        </Grid>

        {/* Recent Analyses */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Analyses
            </Typography>
            <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
              {stats?.recent_analyses && stats.recent_analyses.length > 0 ? (
                stats.recent_analyses.map((analysis) => (
                <Box
                  key={analysis.firmware_id}
                  sx={{
                    p: 2,
                    mb: 1,
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    cursor: 'pointer',
                    '&:hover': { backgroundColor: 'action.hover' },
                  }}
                  onClick={() => navigate(`/analysis/${analysis.firmware_id}`)}
                >
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="subtitle1">{analysis.file_name}</Typography>
                      <Typography variant="caption" color="textSecondary">
                        {new Date(analysis.date).toLocaleString()}
                      </Typography>
                    </Box>
                    <Box>
                      {analysis.is_tampered === true ? (
                        <Chip
                          label="Tampered"
                          color="error"
                          size="small"
                          sx={{ mr: 1 }}
                        />
                      ) : analysis.is_tampered === false ? (
                        <Chip
                          label="Clean"
                          color="success"
                          size="small"
                          sx={{ mr: 1 }}
                        />
                      ) : (
                        <Chip
                          label="Pending"
                          color="warning"
                          size="small"
                          sx={{ mr: 1 }}
                        />
                      )}
                      <Typography variant="body2" color="textSecondary">
                        {analysis.probability !== null && analysis.probability !== undefined
                          ? `${(analysis.probability * 100).toFixed(1)}% confidence`
                          : 'Analysis pending'}
                      </Typography>
                    </Box>
                  </Box>
                </Box>
              ))
              ) : (
                <Typography variant="body2" color="textSecondary" sx={{ p: 2, textAlign: 'center' }}>
                  No analyses yet. Upload firmware to get started.
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Dashboard


