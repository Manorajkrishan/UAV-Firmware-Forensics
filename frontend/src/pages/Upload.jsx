import React, { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Checkbox,
  FormControlLabel,
  Grid,
} from '@mui/material'
import { CloudUpload as UploadIcon } from '@mui/icons-material'
import { useDropzone } from 'react-dropzone'
import { uploadFirmware, analyzeFirmware, uploadAndTrain } from '../services/api'
import { useNavigate } from 'react-router-dom'

function Upload() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [modelPreference, setModelPreference] = useState('ensemble')
  const [trainModels, setTrainModels] = useState(false)
  const [training, setTraining] = useState(false)
  const [trainingResults, setTrainingResults] = useState(null)
  const navigate = useNavigate()

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
      'application/octet-stream': ['.bin', '.hex', '.elf'],
      'application/x-executable': ['.bin', '.elf'],
    },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0])
        setError(null)
      }
    },
    maxFiles: 1,
  })

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    try {
      setUploading(true)
      setError(null)
      setSuccess(null)
      setTrainingResults(null)

      if (trainModels) {
        // Upload, train, and analyze
        setTraining(true)
        const result = await uploadAndTrain(file, true)
        setTrainingResults(result.training)
        setSuccess(`Dataset uploaded, models trained, and analyzed! Firmware ID: ${result.firmware_id}`)
        
        // Navigate to analysis if available
        if (result.analysis) {
          setTimeout(() => {
            navigate(`/analysis/${result.firmware_id}`)
          }, 2000)
        }
      } else {
        // Just upload and analyze
        const uploadResult = await uploadFirmware(file)
        setSuccess(`File uploaded successfully! Firmware ID: ${uploadResult.firmware_id}`)

        // Automatically analyze
        setAnalyzing(true)
        await analyzeFirmware(uploadResult.firmware_id, modelPreference)

        // Navigate to analysis page
        navigate(`/analysis/${uploadResult.firmware_id}`)
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Upload failed')
    } finally {
      setUploading(false)
      setAnalyzing(false)
      setTraining(false)
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Upload Firmware for Analysis
      </Typography>

      <Paper sx={{ p: 4, mt: 3 }}>
        <FormControlLabel
          control={
            <Checkbox
              checked={trainModels}
              onChange={(e) => setTrainModels(e.target.checked)}
              disabled={uploading || analyzing || training}
            />
          }
          label="Train models on this dataset (will retrain all models and analyze)"
          sx={{ mb: 3 }}
        />

        {!trainModels && (
          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel>ML Model Preference</InputLabel>
            <Select
              value={modelPreference}
              label="ML Model Preference"
              onChange={(e) => setModelPreference(e.target.value)}
            >
              <MenuItem value="ensemble">Ensemble (Recommended)</MenuItem>
              <MenuItem value="random_forest">Random Forest</MenuItem>
              <MenuItem value="lstm">LSTM</MenuItem>
              <MenuItem value="autoencoder">Autoencoder</MenuItem>
              <MenuItem value="isolation_forest">Isolation Forest</MenuItem>
            </Select>
          </FormControl>
        )}

        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.300',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            cursor: 'pointer',
            backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
            mb: 3,
          }}
        >
          <input {...getInputProps()} />
          <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            {isDragActive ? 'Drop the file here' : 'Drag & drop a firmware file here'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Supported formats: CSV, BIN, HEX, ELF
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
            Binary files (.bin, .hex, .elf) will be automatically converted to CSV
          </Typography>
          {file && (
            <Typography variant="body1" sx={{ mt: 2 }}>
              Selected: {file.name}
            </Typography>
          )}
        </Box>

        <Button
          variant="contained"
          size="large"
          fullWidth
          onClick={handleUpload}
          disabled={!file || uploading || analyzing || training}
          startIcon={uploading || analyzing || training ? <CircularProgress size={20} /> : <UploadIcon />}
        >
          {uploading
            ? 'Uploading...'
            : training
            ? 'Training Models...'
            : analyzing
            ? 'Analyzing...'
            : trainModels
            ? 'Upload, Train & Analyze'
            : 'Upload & Analyze Firmware'}
        </Button>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mt: 2 }}>
            {success}
          </Alert>
        )}

        {trainingResults && (
          <Card sx={{ mt: 2, bgcolor: 'info.light' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Results
              </Typography>
              <Grid container spacing={2}>
                {Object.entries(trainingResults).map(([model, accuracy]) => (
                  <Grid item xs={6} sm={4} key={model}>
                    <Typography variant="body2">
                      <strong>{model.replace('_', ' ').toUpperCase()}:</strong>{' '}
                      {accuracy !== null ? `${(accuracy * 100).toFixed(2)}%` : 'Failed'}
                    </Typography>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        )}
      </Paper>

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            File Requirements
          </Typography>
          <Typography variant="body2" component="div">
            <ul>
              <li><strong>Supported formats:</strong> CSV, BIN (binary firmware), HEX (Intel HEX), ELF (executable)</li>
              <li><strong>Automatic conversion:</strong> Binary files (.bin, .hex, .elf) are automatically parsed and converted to CSV</li>
              <li><strong>CSV format:</strong> If uploading CSV, required columns: entropy_score, is_signed, boot_time_ms, emulated_syscalls</li>
              <li><strong>For training:</strong> CSV must include target column: clean_label, is_tampered, label, or target</li>
              <li><strong>Features extracted:</strong> Entropy, signatures, strings, IPs, URLs, crypto functions, sections</li>
            </ul>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  )
}

export default Upload



