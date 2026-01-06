# 🖥️ Complete Desktop Application Guide

## ✅ Your System is Now a Desktop App!

Your Drone Firmware Tampering Detection system has been converted into a **standalone desktop application** with a beautiful, modern GUI.

## 🎨 Features

### ✨ Modern GUI Interface
- **Beautiful Design**: Gradient backgrounds, smooth animations, modern UI
- **Three Main Tabs**:
  - 📤 **Upload & Analyze**: Drag-and-drop file upload, real-time analysis
  - 📊 **Dashboard**: Statistics, recent analyses, system status
  - 📜 **History**: View all past analyses

### 🚀 Key Features
- ✅ **Auto-starts Backend**: Python backend starts automatically
- ✅ **File Upload**: Drag-and-drop or click to browse
- ✅ **Real-time Progress**: Progress bar during analysis
- ✅ **Multiple ML Models**: Choose from 5 different models
- ✅ **Visual Results**: Color-coded results (red=tampered, green=clean)
- ✅ **Dashboard Stats**: Live statistics and recent analyses
- ✅ **System Status**: Backend health monitoring

## 📦 Installation

### Step 1: Install Dependencies

**Windows:**
```bash
cd desktop
npm install
```

**Linux/Mac:**
```bash
cd desktop
npm install
```

This installs:
- `electron` - Desktop app framework
- `electron-builder` - For packaging the app

### Step 2: Install Backend Dependencies

```bash
cd ../backend
pip install -r requirements.txt
```

### Step 3: Run the Desktop App

**Windows:**
```bash
cd desktop
start.bat
```

**Linux/Mac:**
```bash
cd desktop
chmod +x start.sh
./start.sh
```

**Or manually:**
```bash
cd desktop
npm start
```

## 🎯 How It Works

1. **Desktop App Starts** → Electron window opens
2. **Backend Auto-Starts** → Python backend starts automatically on port 8000
3. **GUI Loads** → Beautiful standalone HTML interface loads
4. **Ready to Use** → Upload files and analyze!

## 📁 Project Structure

```
Freed/
├── desktop/
│   ├── main.js          # Electron main process (starts backend + window)
│   ├── preload.js       # Security bridge
│   ├── index.html       # Standalone GUI (NEW!)
│   ├── package.json     # Electron dependencies
│   ├── start.bat        # Windows startup script
│   └── start.sh         # Linux/Mac startup script
├── backend/
│   └── main.py          # Python backend (auto-started)
└── frontend/            # React frontend (optional, not needed for desktop)
```

## 🎨 GUI Screenshots

### Upload & Analyze Tab
- Drag-and-drop file upload area
- Model selector dropdown
- Real-time progress bar
- Detailed analysis results with color coding

### Dashboard Tab
- Total analyses count
- Tampered vs Clean statistics
- Recent analyses list
- Auto-refresh capability

### History Tab
- Complete analysis history
- Filter by status
- View all past uploads

## 🔧 Building for Distribution

### Create Windows Installer (.exe)

```bash
cd desktop
npm run build
```

Output: `desktop/dist-electron/Drone Firmware Tampering Detection Setup x.x.x.exe`

### Create Mac App (.dmg)

```bash
cd desktop
npm run build
```

Output: `desktop/dist-electron/Drone Firmware Tampering Detection-x.x.x.dmg`

### Create Linux App (.AppImage)

```bash
cd desktop
npm run build
```

Output: `desktop/dist-electron/Drone Firmware Tampering Detection-x.x.x.AppImage`

## 🎯 Usage Guide

### 1. Upload a File

**Method 1: Drag and Drop**
- Drag a firmware file (CSV, BIN, HEX, ELF) onto the upload area
- File is automatically selected

**Method 2: Click to Browse**
- Click the upload area
- Select file from file dialog

### 2. Select ML Model

Choose from:
- **Ensemble** (Recommended) - Combines all models
- **Random Forest** - Tree-based classifier
- **Isolation Forest** - Anomaly detection
- **LSTM** - Deep learning model
- **Autoencoder** - Neural network

### 3. Analyze

- Click "🔍 Analyze Firmware" button
- Watch progress bar
- View results when complete

### 4. View Results

Results show:
- **Status**: Tampered (red) or Clean (green)
- **Probability**: Tampering probability percentage
- **Model Used**: Which ML model analyzed it
- **Confidence**: Analysis confidence score
- **Severity**: Risk level (Low, Medium, High, Critical)
- **Recommendations**: Action items

### 5. Check Dashboard

- Switch to "📊 Dashboard" tab
- View statistics:
  - Total analyses
  - Tampered count
  - Clean count
  - Recent analyses

### 6. View History

- Switch to "📜 History" tab
- See all past analyses
- Filter and review previous results

## 🛠️ Troubleshooting

### Backend Won't Start

**Problem**: Backend status shows "Offline"

**Solutions**:
1. Check Python is installed: `python --version`
2. Install backend dependencies: `cd backend && pip install -r requirements.txt`
3. Check port 8000 is not in use
4. Check backend logs in terminal

### App Won't Launch

**Problem**: Nothing happens when running `npm start`

**Solutions**:
1. Install dependencies: `npm install`
2. Check Node.js version: `node --version` (should be 16+)
3. Check for errors in terminal
4. Try: `npm run dev` for development mode

### File Upload Fails

**Problem**: "Upload failed" error

**Solutions**:
1. Check backend is running (green dot in status bar)
2. Check file format (CSV, BIN, HEX, ELF)
3. Check file size (not too large)
4. Check backend logs

### Analysis Fails

**Problem**: "Analysis failed" error

**Solutions**:
1. Check backend logs for details
2. Ensure ML models are loaded (check status bar)
3. Verify file has required columns (entropy_score, is_signed, etc.)
4. Try different ML model

## 🎨 Customization

### Change App Icon

1. Create icons:
   - `desktop/assets/icon.ico` (Windows, 256x256)
   - `desktop/assets/icon.icns` (Mac, 512x512)
   - `desktop/assets/icon.png` (Linux, 512x512)

2. Icons are automatically used when building

### Change Colors

Edit `desktop/index.html`:
- Find CSS gradient: `background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);`
- Change colors to your preference

### Change Window Size

Edit `desktop/main.js`:
```javascript
mainWindow = new BrowserWindow({
  width: 1400,  // Change width
  height: 900,  // Change height
  ...
})
```

## 📝 Development Mode

Run with developer tools:

```bash
cd desktop
npm run dev
```

This opens DevTools for debugging.

## 🔒 Security

- **Context Isolation**: Enabled (secure)
- **Node Integration**: Disabled (secure)
- **Preload Script**: Used for safe IPC communication

## 🚀 Advanced Features

### Auto-Update (Future)
- Can add auto-update functionality
- Uses `electron-updater`

### System Tray (Future)
- Minimize to system tray
- Background operation

### Notifications (Future)
- Desktop notifications for analysis complete
- Alert for tampered firmware detected

## ✅ What's Included

- ✅ Standalone HTML GUI (no React needed)
- ✅ Auto-starting backend
- ✅ File upload with drag-and-drop
- ✅ Real-time progress tracking
- ✅ Multiple ML model support
- ✅ Dashboard with statistics
- ✅ Analysis history
- ✅ Beautiful modern UI
- ✅ Cross-platform support (Windows, Mac, Linux)
- ✅ Packaging scripts for distribution

## 🎉 You're Ready!

Your desktop application is complete and ready to use!

1. **Run**: `cd desktop && npm start`
2. **Upload**: Drag a file or click to browse
3. **Analyze**: Click analyze button
4. **View Results**: See tampering detection results

Enjoy your desktop application! 🚀


