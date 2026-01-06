import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  CircularProgress,
  Alert,
  IconButton,
  TextField,
  Button,
  Tooltip,
} from '@mui/material'
import { Visibility as ViewIcon, Refresh as RefreshIcon } from '@mui/icons-material'
import { getAllAnalyses, retryAnalysis } from '../services/api'

function History() {
  const [analyses, setAnalyses] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [retrying, setRetrying] = useState({})
  const navigate = useNavigate()

  useEffect(() => {
    loadAnalyses()
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadAnalyses, 30000)
    return () => clearInterval(interval)
  }, [])

  const loadAnalyses = async () => {
    try {
      setLoading(true)
      const data = await getAllAnalyses()
      setAnalyses(data)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRetry = async (firmwareId) => {
    try {
      setRetrying({ ...retrying, [firmwareId]: true })
      setError(null) // Clear previous errors
      const result = await retryAnalysis(firmwareId, 'ensemble')
      console.log('Retry successful:', result)
      // Reload analyses after retry
      await loadAnalyses()
      // Show success message briefly
      setTimeout(() => setError(null), 3000)
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'Retry failed'
      console.error('Retry failed:', errorMsg)
      setError(`Retry failed: ${errorMsg}`)
      // Keep error visible for 5 seconds
      setTimeout(() => setError(null), 5000)
    } finally {
      setRetrying({ ...retrying, [firmwareId]: false })
    }
  }

  const filteredAnalyses = analyses.filter((analysis) =>
    analysis.file_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    analysis.firmware_id.toLowerCase().includes(searchTerm.toLowerCase())
  )

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return <Alert severity="error">Error loading history: {error}</Alert>
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Analysis History
      </Typography>

      {error && (
        <Alert 
          severity="error" 
          sx={{ mb: 2 }}
          onClose={() => setError(null)}
        >
          {error}
        </Alert>
      )}

      <TextField
        fullWidth
        label="Search analyses"
        variant="outlined"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        sx={{ mb: 3, mt: 2 }}
      />

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>File Name</TableCell>
              <TableCell>Upload Date</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Result</TableCell>
              <TableCell>Probability</TableCell>
              <TableCell>Model</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredAnalyses.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography color="text.secondary">No analyses found</Typography>
                </TableCell>
              </TableRow>
            ) : (
              filteredAnalyses.map((analysis) => (
                <TableRow key={analysis.firmware_id} hover>
                  <TableCell>{analysis.file_name}</TableCell>
                  <TableCell>
                    {new Date(analysis.upload_date).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={analysis.analysis_status}
                      color={
                        analysis.analysis_status === 'completed' 
                          ? 'success' 
                          : analysis.analysis_status === 'failed'
                          ? 'error'
                          : 'warning'
                      }
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {analysis.is_tampered !== null ? (
                      <Chip
                        label={analysis.is_tampered ? 'Tampered' : 'Clean'}
                        color={analysis.is_tampered ? 'error' : 'success'}
                        size="small"
                      />
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        N/A
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {analysis.tampering_probability !== null
                      ? `${(analysis.tampering_probability * 100).toFixed(1)}%`
                      : 'N/A'}
                  </TableCell>
                  <TableCell>{analysis.model_used || 'N/A'}</TableCell>
                  <TableCell>
                    <Box display="flex" gap={1}>
                      {(analysis.analysis_status === 'pending' || analysis.analysis_status === 'failed') && (
                        <Tooltip title="Retry Analysis">
                          <IconButton
                            size="small"
                            onClick={() => handleRetry(analysis.firmware_id)}
                            disabled={retrying[analysis.firmware_id]}
                            color="primary"
                          >
                            {retrying[analysis.firmware_id] ? (
                              <CircularProgress size={20} />
                            ) : (
                              <RefreshIcon />
                            )}
                          </IconButton>
                        </Tooltip>
                      )}
                      {analysis.analysis_status === 'completed' && (
                        <Tooltip title="View Details">
                          <IconButton
                            size="small"
                            onClick={() => navigate(`/analysis/${analysis.firmware_id}`)}
                          >
                            <ViewIcon />
                          </IconButton>
                        </Tooltip>
                      )}
                    </Box>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  )
}

export default History



