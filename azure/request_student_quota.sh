#!/bin/bash

echo "=== Azure Student GPU Quota Request Guide ==="
echo
echo "1. Open Azure Portal:"
echo "   https://portal.azure.com"
echo
echo "2. Search for 'Quotas' → 'Compute quotas'"
echo
echo "3. Request these quotas (most likely to be approved):"
echo
echo "   Option A - Old but Available (K80 GPU):"
echo "   - Name: Standard NC Family"
echo "   - Current: 6"
echo "   - Request: Keep at 6 (already have quota!)"
echo "   - VM to use: Standard_NC6"
echo
echo "   Option B - Visualization GPU (M60):"
echo "   - Name: Standard NV Family" 
echo "   - Current: 6"
echo "   - Request: Keep at 6 (already have quota!)"
echo "   - VM to use: Standard_NV6"
echo
echo "   Option C - Modern T4 GPU (if need to request):"
echo "   - Name: Standard NCASv3_T4 Family"
echo "   - Current: 0"
echo "   - Request: 4 vCPUs"
echo "   - VM to use: Standard_NC4as_T4_v3"
echo
echo "4. In 'Business Justification' write:"
echo "   'Educational machine learning research project for"
echo "   vision-language model training as part of coursework.'"
echo
echo "5. Submit and wait 1-3 business days"
echo
echo "=== GOOD NEWS: You already have quota for NC and NV families! ==="
echo "You can use these VMs right now without requesting:"
echo "- Standard_NC6 (1x K80 GPU, 6 vCPUs)"
echo "- Standard_NV6 (1x M60 GPU, 6 vCPUs)"