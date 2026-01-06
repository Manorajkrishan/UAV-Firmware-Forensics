import React from 'react'
import {
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Chip
} from '@mui/material'
import { CheckCircle, Cancel, Warning } from '@mui/icons-material'

function EvidencePanel({ analysis }) {
  const sha256Hash = analysis?.sha256_hash || 'N/A'
  const firmwareId = analysis?.firmware_id || 'N/A'
  const fileName = analysis?.file_name || 'N/A'
  const integrityStatus = sha256Hash !== 'N/A' ? 'Verified' : 'Not Available'
  const isTampered = analysis?.is_tampered || false
  const tamperingStatus = analysis?.tampering_status || 'Unknown'
  
  const evidenceData = [
    { label: 'Evidence ID', value: firmwareId },
    { label: 'File Name', value: fileName },
    { label: 'SHA-256 Hash', value: sha256Hash },
    { label: 'File Integrity', value: integrityStatus },
    { label: 'Tampering Status', value: tamperingStatus },
    { label: 'Analysis Date', value: analysis?.timestamp ? new Date(analysis.timestamp).toLocaleString() : 'N/A' }
  ]
  
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Evidence Information
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Cryptographic integrity proof and evidence details
      </Typography>
      
      <TableContainer>
        <Table size="small">
          <TableBody>
            {evidenceData.map((row, index) => (
              <TableRow key={index}>
                <TableCell component="th" scope="row" sx={{ fontWeight: 'bold', width: '40%' }}>
                  {row.label}
                </TableCell>
                <TableCell>
                  {row.label === 'Tampering Status' ? (
                    <Chip
                      label={row.value}
                      color={
                        row.value === 'Tampered' ? 'error' :
                        row.value === 'Suspicious' ? 'warning' :
                        'success'
                      }
                      size="small"
                    />
                  ) : row.label === 'File Integrity' ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {row.value === 'Verified' ? (
                        <CheckCircle color="success" fontSize="small" />
                      ) : (
                        <Warning color="warning" fontSize="small" />
                      )}
                      {row.value}
                    </Box>
                  ) : (
                    <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                      {row.value}
                    </Typography>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      {analysis?.integrity_checks && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Integrity Check Details
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {analysis.integrity_checks.integrity_details?.join(', ') || 'No integrity issues detected'}
          </Typography>
        </Box>
      )}
    </Paper>
  )
}

export default EvidencePanel

