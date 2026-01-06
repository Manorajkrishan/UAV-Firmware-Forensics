import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const uploadFirmware = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const analyzeFirmware = async (firmwareId, modelPreference = 'ensemble') => {
  const response = await api.post('/api/analyze', {
    firmware_id: firmwareId,
    model_preference: modelPreference,
  })
  return response.data
}

export const getAnalysis = async (firmwareId) => {
  const response = await api.get(`/api/analyses/${firmwareId}`)
  return response.data
}

export const getAllAnalyses = async (skip = 0, limit = 100) => {
  try {
    const response = await api.get('/api/analyses', {
      params: { skip, limit },
    })
    return response.data || []
  } catch (error) {
    // Return empty array if API fails (database not available)
    console.warn('Failed to fetch analyses, returning empty array:', error.message)
    return []
  }
}

export const getDashboardStats = async () => {
  try {
    const response = await api.get('/api/dashboard/stats', {
      timeout: 5000, // 5 second timeout
      validateStatus: function (status) {
        // Accept any status code, don't throw error
        return status < 600
      }
    })
    // Always return data, even if status is not 200
    return response.data || {
      total_analyses: 0,
      tampered_count: 0,
      clean_count: 0,
      recent_analyses: []
    }
  } catch (error) {
    // Return default stats if API fails - never throw error
    console.warn('Failed to fetch dashboard stats, using defaults:', error.message)
    return {
      total_analyses: 0,
      tampered_count: 0,
      clean_count: 0,
      recent_analyses: []
    }
  }
}

export const healthCheck = async () => {
  const response = await api.get('/health')
  return response.data
}

export const trainModels = async (datasetPath, testSize = 0.2) => {
  const response = await api.post('/api/train', {
    dataset_path: datasetPath,
    test_size: testSize,
  })
  return response.data
}

export const uploadAndTrain = async (file, train = true) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post(`/api/upload-and-train?train=${train}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const retryAnalysis = async (firmwareId, modelPreference = 'ensemble') => {
  const response = await api.post(`/api/analyses/${firmwareId}/retry?model_preference=${modelPreference}`)
  return response.data
}

export default api


