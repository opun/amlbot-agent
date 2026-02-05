# Deploy on AWS EC2 (Amazon Linux) with Docker Compose

These steps assume you already have a working `docker-compose-remote.yml`, `Caddyfile`, and the required env files: `.api.env` and `.next.env`.

## 1) Launch EC2
1. Create an EC2 instance using **Amazon Linux 2023** (or Amazon Linux 2).
2. Create or select a key pair.
3. Security Group inbound rules:
   - `SSH` on port `22` from your IP
   - `HTTP` on port `80` from anywhere
   - `HTTPS` on port `443` from anywhere
4. Note the instance **Public IPv4** address.

## 2) SSH into the instance
```bash
ssh -i /path/to/key.pem ec2-user@PUBLIC_IP
```

## 3) Update OS packages
Amazon Linux 2023:
```bash
sudo dnf -y update
```
Amazon Linux 2:
```bash
sudo yum -y update
```

## 4) Install Docker
Amazon Linux 2023:
```bash
sudo dnf -y install docker
```
Amazon Linux 2:
```bash
sudo amazon-linux-extras install docker -y
```

Enable and start Docker:
```bash
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker
```

## 5) Install Docker Compose
Preferred (Compose v2 plugin):
Amazon Linux 2023:
```bash
sudo dnf -y install docker-compose-plugin
```
Amazon Linux 2:
```bash
sudo yum -y install docker-compose-plugin
```

If the plugin package is unavailable on your image, install Compose v2 manually:
```bash
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```
Verify:
```bash
docker compose version
```

## 6) Point DNS to your EC2
Create `A` records:
- `stage.aml-agent.vareger.com` -> your EC2 Public IPv4
- `stage.api.aml-agent.vareger.com` -> your EC2 Public IPv4

Wait for DNS to propagate before starting Caddy.

## 7) Login to GitHub Container Registry (private images)
```bash
echo "YOUR_GHCR_TOKEN" | docker login ghcr.io -u YOUR_GHCR_USERNAME --password-stdin
```

## 8) Upload your compose files
From your local machine:
```bash
ssh -i /path/to/key.pem ec2-user@PUBLIC_IP "mkdir -p ~/app"
scp -i /path/to/key.pem docker-compose-remote.yml ec2-user@PUBLIC_IP:~/app/
scp -i /path/to/key.pem Caddyfile ec2-user@PUBLIC_IP:~/app/
scp -i /path/to/key.pem .api.env ec2-user@PUBLIC_IP:~/app/
scp -i /path/to/key.pem .next.env ec2-user@PUBLIC_IP:~/app/
```

Alternatively, clone your repo on the instance and copy the files there.

## Caddyfile (example)
Use this exact file for your domains:
```caddyfile
stage.aml-agent.vareger.com {
  encode gzip
  reverse_proxy frontend:3000
}

stage.api.aml-agent.vareger.com {
  encode gzip
  reverse_proxy api:8000
}
```

## 9) Start the stack
```bash
cd ~/app
# Optional: pull images first
# docker compose -f docker-compose-remote.yml pull

docker compose -f docker-compose-remote.yml up -d
```

## 10) Verify
```bash
docker compose -f docker-compose-remote.yml ps
# Check logs for a service
# docker compose -f docker-compose-remote.yml logs -f <service_name>
```
Visit your apps:
- `https://stage.aml-agent.vareger.com`
- `https://stage.api.aml-agent.vareger.com`

## 11) Updates
```bash
cd ~/app
docker compose -f docker-compose-remote.yml pull
docker compose -f docker-compose-remote.yml up -d
```

## 12) Optional hardening
- `restart: unless-stopped` is already set for `caddy`, `api`, and `frontend` in `docker-compose-remote.yml`.
- Use an Elastic IP for a stable address.
- Put the instance behind an ALB and use HTTPS (ACM certs).
