-- PostgreSQL Database Initialization
-- Drone Firmware Tampering Detection System

-- Note: Database should be created manually or via docker-compose
-- CREATE DATABASE drone_forensics;

-- The FirmwareAnalysis table will be created by SQLAlchemy
-- This file is for reference and manual setup if needed

CREATE TABLE IF NOT EXISTS firmware_analyses (
    id SERIAL PRIMARY KEY,
    firmware_id VARCHAR(255) UNIQUE NOT NULL,
    file_name VARCHAR(255),
    file_hash VARCHAR(255),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_status VARCHAR(50) DEFAULT 'pending',
    is_tampered BOOLEAN DEFAULT FALSE,
    tampering_probability FLOAT,
    model_used VARCHAR(100),
    analysis_results TEXT,
    evidence_path VARCHAR(500)
);

CREATE INDEX IF NOT EXISTS idx_firmware_id ON firmware_analyses(firmware_id);
CREATE INDEX IF NOT EXISTS idx_upload_date ON firmware_analyses(upload_date);
CREATE INDEX IF NOT EXISTS idx_is_tampered ON firmware_analyses(is_tampered);

