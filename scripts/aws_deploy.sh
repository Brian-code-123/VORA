#!/usr/bin/env bash
# Run INSIDE AWS CloudShell (it already has the console login; no keys leave AWS). From the folder holding vora-src.tar.gz:
#   tar xzf vora-src.tar.gz scripts/aws_deploy.sh && bash scripts/aws_deploy.sh <YOUR_PUBLIC_IP> [demo|a1|both]
# demo: t4g.small (arm64, 2 vCPU, 2 GB + 4 GB swap), 30 GB gp3, HTTPS on port 443 for any network, /ws needs an access key (first message), a
#       self-signed certificate (browsers need HTTPS for the microphone). Restarts with the box. No SSH port is opened.
# a1:   a1.xlarge (Graviton1 = 4x Cortex-A72, the Raspberry Pi 4 core) runs scripts/pi_bench.sh in the image, uploads the
#       results to the bucket, and terminates itself; hard stop after 3 h whatever happens.
# Everything is tagged project=vora; scripts/aws_teardown.sh removes it and checks that nothing is left.
set -euo pipefail
MYIP="${1:?usage: aws_deploy.sh <your public IP|any> [demo|a1|both]}"   # kept for the record; the demo is protected by an access key
MODE="${2:-both}"
REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$REGION"
[ -f vora-src.tar.gz ] || { echo "vora-src.tar.gz not in $(pwd)" >&2; exit 1; }

ACC=$(aws sts get-caller-identity --query Account --output text)
B="vora-deploy-${ACC}"
if ! aws s3api head-bucket --bucket "$B" 2>/dev/null; then
  aws s3api create-bucket --bucket "$B" >/dev/null
  aws s3api put-public-access-block --bucket "$B" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi
aws s3 cp vora-src.tar.gz "s3://$B/src.tar.gz" --only-show-errors
SRC_URL=$(aws s3 presign "s3://$B/src.tar.gz" --expires-in 43200)
PUT_URL=$(python3 -c "import boto3,sys; print(boto3.client('s3', region_name='$REGION').generate_presigned_url('put_object', Params={'Bucket': '$B', 'Key': 'results/a1-results.tar.gz'}, ExpiresIn=43200))")
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 --query Parameter.Value --output text)
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)

sg() {   # $1 name, $2 "open443"|"none"
  local id
  id=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$1" Name=vpc-id,Values="$VPC" --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$id" = "None" ]; then
    id=$(aws ec2 create-security-group --group-name "$1" --description "VORA ($1)" --vpc-id "$VPC" \
         --tag-specifications "ResourceType=security-group,Tags=[{Key=project,Value=vora}]" --query GroupId --output text)
  fi
  if [ "$2" = "open443" ]; then   # reachable from any network on the standard HTTPS port; /ws requires the access key
    aws ec2 revoke-security-group-ingress --group-id "$id" --ip-permissions "$(aws ec2 describe-security-groups --group-ids "$id" \
      --query 'SecurityGroups[0].IpPermissions' --output json)" >/dev/null 2>&1 || true        # drop old 8000 rules
    aws ec2 authorize-security-group-ingress --group-id "$id" --protocol tcp --port 443 --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
  fi
  echo "$id"
}

common_userdata() {   # shared boot steps: swap, docker, source, image build. Progress goes to the serial console.
cat <<EOF
#!/bin/bash
exec > >(tee /var/log/vora-boot.log | logger -t vora -s 2>/dev/console) 2>&1
say() { echo "VORA: \$*" > /dev/console; echo "VORA: \$*"; }
say start \$(date -u +%FT%TZ)
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
dnf install -y docker tar >/dev/null && systemctl enable --now docker
mkdir -p /opt/vora && curl -fsS "$SRC_URL" | tar xz -C /opt/vora
cd /opt/vora
t0=\$(date +%s); docker build -f docker/Dockerfile -t vora . > /var/log/vora-build.log 2>&1 || { say BUILD FAILED; tail -30 /var/log/vora-build.log > /dev/console; exit 1; }
say built in \$(( \$(date +%s)-t0 )) s, image \$(docker images vora --format '{{.Size}}')
EOF
}

launch() {   # $1 type, $2 name, $3 sg, $4 userdata file, $5 extra args
  aws ec2 run-instances --image-id "$AMI" --instance-type "$1" --security-group-ids "$3" \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}' \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled --user-data "file://$4" $5 \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$2},{Key=project,Value=vora}]" \
                         "ResourceType=volume,Tags=[{Key=project,Value=vora}]" \
    --query 'Instances[0].InstanceId' --output text
}

if [ "$MODE" = demo ] || [ "$MODE" = both ]; then
  KEY=$(openssl rand -hex 12)          # shown once below, on YOUR screen only; also inside the instance user-data
  old=$(aws ec2 describe-instances --filters Name=tag:Name,Values=vora-demo Name=instance-state-name,Values=pending,running,stopping,stopped \
        --query 'Reservations[].Instances[].InstanceId' --output text)
  [ -n "$old" ] && aws ec2 terminate-instances --instance-ids $old >/dev/null && echo "replacing old demo: $old"
  { common_userdata; echo "KEY=$KEY"; cat <<'EOF'
ip link set dev "$(ip -o route get 1.1.1.1 | awk '{print $5}')" mtu 1400   # VPN paths dropped full-size TLS packets (handshake stalled)
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
mkdir -p /opt/vora/certs && openssl req -x509 -newkey rsa:2048 -nodes -days 60 -keyout /opt/vora/certs/vora.key \
  -out /opt/vora/certs/vora.crt -subj "/CN=vora-demo" -addext "subjectAltName=IP:${IP}" 2>/dev/null
chmod 644 /opt/vora/certs/vora.key
docker run -d --name vora --restart unless-stopped -p 443:8000 -v /opt/vora/certs:/certs:ro -v /opt/vora/scripts:/opt/smoke:ro \
  -e VORA_SSL_CERT=/certs/vora.crt -e VORA_SSL_KEY=/certs/vora.key -e VORA_ACCESS_KEY="$KEY" -e VORA_LLM_THREADS=2 -e VORA_ASR_THREADS=1 vora
for i in $(seq 1 120); do curl -kfs https://127.0.0.1/health >/dev/null && break; sleep 5; done
say READY https://${IP}/ after $((i*5)) s
for f in q_warranty_en.wav:en real/zh_balance.wav:zh; do    # self-test through the real socket, with and without the key
  say "smoke ${f}: $(docker exec -e VORA_KEY="$KEY" vora python /opt/smoke/ws_smoke.py wss://127.0.0.1:8000/ws /app/client/samples/${f%%:*} --lang ${f##*:} --insecure 2>&1 | tail -1)"
done
say "smoke no key: $(docker exec vora python /opt/smoke/ws_smoke.py wss://127.0.0.1:8000/ws /app/client/samples/q_warranty_en.wav --insecure --timeout 5 2>&1 | tail -1)"
say "uid: $(docker exec vora id -u)"      # expected 10001: the container must not run as root
EOF
  } > /tmp/ud-demo.sh
  DEMO_ID=$(launch t4g.small vora-demo "$(sg vora-demo-sg open443)" /tmp/ud-demo.sh "")
  aws ec2 wait instance-running --instance-ids "$DEMO_ID"
  DIP=$(aws ec2 describe-instances --instance-ids "$DEMO_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "demo instance: $DEMO_ID  (ready in ~10 min)"
  echo "OPEN: https://${DIP}/#key=${KEY}"   # a fragment is never sent to the server, so the key stays out of its access log
fi

if [ "$MODE" = a1 ] || [ "$MODE" = both ]; then
  { cat <<'EOF'
#!/bin/bash
shutdown -h +180 "VORA a1 hard stop"
EOF
  common_userdata | tail -n +2; cat <<EOF
mkdir -p /opt/vora/results-pi && lscpu > /opt/vora/results-pi/lscpu.txt
t0=\$(date +%s); docker run -d --name vora -p 127.0.0.1:8000:8000 -v /opt/vora/scripts:/opt/smoke:ro vora
for i in \$(seq 1 120); do curl -fs http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
echo "ready_s=\$(( \$(date +%s)-t0 ))" > /opt/vora/results-pi/ready.txt
for f in q_warranty_en.wav:en q_warranty_zh.wav:zh real/zh_balance.wav:zh real/en_us_balance.wav:en; do
  docker exec vora python /opt/smoke/ws_smoke.py ws://127.0.0.1:8000/ws /app/client/samples/\${f%%:*} --lang \${f##*:} >> /opt/vora/results-pi/ws_smoke.txt 2>&1 || true
done
docker rm -f vora >/dev/null
docker run --rm -v /opt/vora/results-pi:/app/results -v /opt/vora/scripts:/app/scripts:ro vora bash scripts/pi_bench.sh > /opt/vora/results-pi/pi_bench.log 2>&1 || true
say "\$(grep -m1 -i 'model name' /opt/vora/results-pi/lscpu.txt)"
say "\$(cat /opt/vora/results-pi/ready.txt)"; while read -r l; do say "smoke \$l"; done < /opt/vora/results-pi/ws_smoke.txt
[ -f /opt/vora/results-pi/pi_tts.json ] && say "tts \$(tr -d '\n' < /opt/vora/results-pi/pi_tts.json)"
tar czf /tmp/a1-results.tar.gz -C /opt/vora results-pi && curl -fsS -X PUT --upload-file /tmp/a1-results.tar.gz "$PUT_URL" && say uploaded
say done; shutdown -h now
EOF
  } > /tmp/ud-a1.sh
  if A1_ID=$(launch a1.xlarge vora-a1 "$(sg vora-a1-sg none)" /tmp/ud-a1.sh "--instance-initiated-shutdown-behavior terminate"); then
    echo "a1 instance: $A1_ID"
  else
    echo "a1.xlarge launch refused (the free account plan may not allow it): Pi-class run stays UNVERIFIED"
  fi
fi
echo "bucket: $B"
echo "progress: aws ec2 get-console-output --latest --instance-id <id> --query Output --output text | grep VORA:"
