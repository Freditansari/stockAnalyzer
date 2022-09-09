docker build --tag stockanalyzer:latest .
sudo docker run --name stockanalyzerServer -d -p 81:5000 stockanalyzer