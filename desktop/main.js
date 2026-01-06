const { app, BrowserWindow, ipcMain, dialog } = require('electron')
const path = require('path')
const { spawn, exec } = require('child_process')
const fs = require('fs')

// Keep a global reference of the window object
let mainWindow
let backendProcess = null
let frontendProcess = null

// Backend server configuration
const BACKEND_PORT = 8000
const BACKEND_HOST = 'localhost'

// Path to backend
const BACKEND_PATH = path.join(__dirname, '..', 'backend', 'main.py')
const PYTHON_EXECUTABLE = process.platform === 'win32' ? 'python' : 'python3'

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    icon: path.join(__dirname, 'assets', 'icon.png'), // Optional: add icon
    titleBarStyle: 'default',
    show: false, // Don't show until ready
  })

  // Load the React frontend with all features (graphs, charts, reports)
  const isDev = process.argv.includes('--dev') || process.env.NODE_ENV === 'development'
  
  if (isDev) {
    // Development: load from Vite dev server (will be started separately)
    console.log('Loading React frontend from Vite dev server...')
    mainWindow.loadURL('http://localhost:3000')
    mainWindow.webContents.openDevTools()
  } else {
    // Production: load from built files
    const distPath = path.join(__dirname, '..', 'frontend', 'dist', 'index.html')
    if (fs.existsSync(distPath)) {
      console.log('Loading React frontend from built files...')
      mainWindow.loadFile(distPath)
    } else {
      // Build frontend if not built
      console.warn('Frontend not built. Building now...')
      dialog.showMessageBox(mainWindow, {
        type: 'info',
        title: 'Building Frontend',
        message: 'The frontend needs to be built first. Please run: cd frontend && npm run build'
      })
      // Try to build automatically
      const frontendDir = path.join(__dirname, '..', 'frontend')
      if (fs.existsSync(path.join(frontendDir, 'package.json'))) {
        console.log('Attempting to build frontend...')
        exec('npm run build', { cwd: frontendDir }, (error) => {
          if (error) {
            console.error('Build failed:', error)
            dialog.showErrorBox(
              'Build Failed',
              'Could not build frontend. Please run: cd frontend && npm run build'
            )
          } else {
            console.log('Frontend built successfully, reloading...')
            mainWindow.loadFile(distPath)
          }
        })
      } else {
        dialog.showErrorBox(
          'Frontend Not Found',
          'Could not find the React frontend. Please ensure frontend directory exists.'
        )
      }
    }
  }

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show()
  })

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

// Check if Python dependencies are installed
function checkPythonDependencies() {
  return new Promise((resolve, reject) => {
    console.log('Checking Python dependencies...')
    
    // Try to import fastapi to check if dependencies are installed
    exec(`"${PYTHON_EXECUTABLE}" -c "import fastapi; print('OK')"`, (error, stdout, stderr) => {
      if (error) {
        console.error('Python dependencies check failed:', error.message)
        console.error('Please install backend dependencies:')
        console.error(`  cd backend && ${PYTHON_EXECUTABLE} -m pip install -r requirements.txt`)
        reject(new Error('Python dependencies not installed. Please run: cd backend && pip install -r requirements.txt'))
      } else {
        console.log('✓ Python dependencies OK')
        resolve()
      }
    })
  })
}

// Start backend server
function startBackend() {
  return new Promise((resolve, reject) => {
    console.log('Starting backend server...')
    console.log(`Python: ${PYTHON_EXECUTABLE}`)
    console.log(`Backend: ${BACKEND_PATH}`)

    // Check if backend file exists
    if (!fs.existsSync(BACKEND_PATH)) {
      console.error(`Backend file not found: ${BACKEND_PATH}`)
      reject(new Error(`Backend file not found: ${BACKEND_PATH}`))
      return
    }

    // Start Python backend
    backendProcess = spawn(PYTHON_EXECUTABLE, [BACKEND_PATH], {
      cwd: path.join(__dirname, '..', 'backend'),
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: process.platform === 'win32',
    })

    let backendReady = false
    let healthCheckInterval = null
    
    // Start periodic health checks after 15 seconds (give backend time to load models)
    const startHealthChecks = setTimeout(() => {
      if (!backendReady) {
        healthCheckInterval = setInterval(() => {
          if (!backendReady) {
            const http = require('http')
            const req = http.get(`http://${BACKEND_HOST}:${BACKEND_PORT}/health`, (res) => {
              if (res.statusCode === 200) {
                console.log('[OK] Backend health check passed - server is running')
                backendReady = true
                clearTimeout(timeout)
                if (healthCheckInterval) clearInterval(healthCheckInterval)
                resolve()
              }
            })
            req.on('error', () => {
              // Health check failed, will try again next interval
            })
            req.setTimeout(2000, () => {
              req.destroy()
            })
          } else {
            if (healthCheckInterval) clearInterval(healthCheckInterval)
          }
        }, 2000) // Check every 2 seconds
      }
    }, 15000) // Start health checks after 15 seconds
    
    const timeout = setTimeout(() => {
      clearTimeout(startHealthChecks)
      if (healthCheckInterval) clearInterval(healthCheckInterval)
      if (!backendReady) {
        console.error('Backend startup timeout')
        // Final health check attempt
        const http = require('http')
        const req = http.get(`http://${BACKEND_HOST}:${BACKEND_PORT}/health`, (res) => {
          if (res.statusCode === 200) {
            console.log('[OK] Backend is actually running, timeout was false alarm')
            backendReady = true
            resolve()
          } else {
            reject(new Error('Backend startup timeout - health check failed'))
          }
        })
        req.on('error', () => {
          reject(new Error('Backend startup timeout - server not responding'))
        })
        req.setTimeout(2000, () => {
          req.destroy()
          reject(new Error('Backend startup timeout - health check timed out'))
        })
      }
    }, 60000) // 60 second timeout (increased for model loading)

    // Helper function to check if backend is ready
    const checkBackendReady = (output) => {
      const readyMessages = [
        'Uvicorn running',
        'Application startup complete',
        'Started server process',
        'INFO:     Uvicorn running',
        'INFO:     Application startup complete'
      ]
      
      return readyMessages.some(msg => output.includes(msg))
    }

    // Handle backend output
    backendProcess.stdout.on('data', (data) => {
      const output = data.toString()
      console.log(`[Backend] ${output}`)
      
      // Check if backend is ready
      if (checkBackendReady(output) && !backendReady) {
        backendReady = true
        clearTimeout(timeout)
        clearTimeout(startHealthChecks)
        if (healthCheckInterval) clearInterval(healthCheckInterval)
        console.log('[OK] Backend server started successfully')
        // Give it a moment to fully initialize
        setTimeout(() => resolve(), 1000)
      }
    })

    backendProcess.stderr.on('data', (data) => {
      const output = data.toString()
      
      // Filter out TensorFlow/TensorFlow info messages (they're not errors)
      const isInfoMessage = output.includes('tensorflow') || 
                           output.includes('oneDNN') || 
                           output.includes('FutureWarning') || 
                           output.includes('InconsistentVersionWarning') ||
                           output.includes('WARNING:absl') ||
                           output.includes('INFO:') ||
                           output.includes('Started server process') ||
                           output.includes('Uvicorn running') ||
                           output.includes('Application startup complete')
      
      if (isInfoMessage) {
        // Log as info instead of error
        console.log(`[Backend Info] ${output.trim()}`)
      } else {
        console.error(`[Backend Error] ${output}`)
      }
      
      // Also check stderr for ready messages (Uvicorn outputs to stderr on Windows)
      if (checkBackendReady(output) && !backendReady) {
        backendReady = true
        clearTimeout(timeout)
        clearTimeout(startHealthChecks)
        if (healthCheckInterval) clearInterval(healthCheckInterval)
        console.log('[OK] Backend server started successfully (detected in stderr)')
        setTimeout(() => resolve(), 1000)
      }
    })

    backendProcess.on('error', (error) => {
      console.error('Failed to start backend:', error)
      clearTimeout(timeout)
      clearTimeout(startHealthChecks)
      if (healthCheckInterval) clearInterval(healthCheckInterval)
      reject(error)
    })

    backendProcess.on('exit', (code) => {
      clearTimeout(startHealthChecks)
      if (healthCheckInterval) clearInterval(healthCheckInterval)
      console.log(`Backend process exited with code ${code}`)
      if (code !== 0 && code !== null) {
        console.error('Backend crashed!')
      }
    })
  })
}

// Stop backend server
function stopBackend() {
  if (backendProcess) {
    console.log('Stopping backend server...')
    backendProcess.kill()
    backendProcess = null
  }
}

// App event handlers
app.whenReady().then(async () => {
  try {
    // Check Python dependencies first
    try {
      await checkPythonDependencies()
    } catch (depError) {
      dialog.showErrorBox(
        'Missing Dependencies',
        `Python dependencies are not installed.\n\n` +
        `Please run:\n` +
        `  cd backend\n` +
        `  pip install -r requirements.txt\n\n` +
        `Error: ${depError.message}`
      )
      app.quit()
      return
    }
    
    // Start backend
    await startBackend()
    
    // Wait a bit for backend to be fully ready
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Create window
    createWindow()

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createWindow()
      }
    })
  } catch (error) {
    console.error('Failed to start application:', error)
    dialog.showErrorBox(
      'Startup Error',
      `Failed to start backend server:\n${error.message}\n\nPlease ensure Python is installed and backend dependencies are installed.`
    )
    app.quit()
  }
})

app.on('window-all-closed', () => {
  // Stop backend when all windows are closed
  stopBackend()
  
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopBackend()
})

// IPC handlers for file dialogs
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Firmware Files', extensions: ['csv', 'bin', 'hex', 'elf'] },
      { name: 'CSV Files', extensions: ['csv'] },
      { name: 'Binary Files', extensions: ['bin', 'hex', 'elf'] },
      { name: 'All Files', extensions: ['*'] },
    ],
    title: 'Select Firmware File to Analyze'
  })
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0]
  }
  return null
})

// Handle save file dialog
ipcMain.handle('save-file', async (event, defaultPath, filters) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: defaultPath,
    filters: filters || [
      { name: 'All Files', extensions: ['*'] }
    ]
  })
  
  if (!result.canceled && result.filePath) {
    return result.filePath
  }
  return null
})

// Health check for backend
ipcMain.handle('check-backend', async () => {
  try {
    const http = require('http')
    return new Promise((resolve) => {
      const req = http.get(`http://${BACKEND_HOST}:${BACKEND_PORT}/health`, (res) => {
        let data = ''
        res.on('data', (chunk) => { data += chunk })
        res.on('end', () => {
          try {
            resolve({ status: res.statusCode, data: JSON.parse(data) })
          } catch {
            resolve({ status: res.statusCode, data: {} })
          }
        })
      })
      req.on('error', () => resolve({ status: 0, error: 'Connection failed' }))
      req.setTimeout(2000, () => {
        req.destroy()
        resolve({ status: 0, error: 'Timeout' })
      })
    })
  } catch (error) {
    return { status: 0, error: error.message }
  }
})

