#!/bin/bash

set -e

echo "Building dicom-listener..."
cd dicom-listener
docker build -t dicom-listener .

echo "Building ai-inference..."
cd ..
docker build -f docker/Dockerfile -t ai-inference .

echo "Stopping old containers..."
docker stop ai-inference dicom-listener 2>/dev/null || true
docker rm ai-inference dicom-listener 2>/dev/null || true

echo "Creating docker network if needed..."
docker network create dicom-net 2>/dev/null || true

echo "Starting ai-inference..."
docker run -d --name ai-inference --network dicom-net --gpus all -p 5000:5000 ai-inference

echo "Starting dicom-listener..."
docker run -d --name dicom-listener --network dicom-net -p 11112:11112 dicom-listener

timeout /t 10

@REM echo "Sending test DICOM file..."
@REM storescu localhost 11112 data/inference/images/local/1.2.826.0.1.3680043.8.498.10000158111648211433066933766838827405.dcm -aec CARPLSTORESCP

echo "Sending several test DICOM files..."
storescu localhost 11112 data/inference/images/local/1.2.826.0.1.3680043.8.498.10000158111648211433066933766838827405.dcm -aec CARPLSTORESCP
storescu localhost 11112 data/inference/images/local/1.2.826.0.1.3680043.8.498.10000523726224485779847156775441329333.dcm -aec CARPLSTORESCP
storescu localhost 11112 data/inference/images/local/1.2.826.0.1.3680043.8.498.10000961223376763459055848908430478242.dcm -aec CARPLSTORESCP



echo "Checking Docker container status..."
docker ps

echo "Done."