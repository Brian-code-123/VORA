# Pi-class run on AWS a1.xlarge (Cortex-A72) — runbook

Only with the owner's explicit yes, with the price and region stated first. No credentials are stored in the repo; the
throwaway SSH key lives in a temporary folder and is deleted at the end.

1. **Check the offering** (previous-generation type): `ec2 DescribeInstanceTypeOfferings` for `a1.xlarge` in
   us-east-1 / us-east-2 / us-west-2 / eu-west-1. Not offered → stop, G7 `pi_class_measured` stays unverified.
2. **Price**: on-demand a1.xlarge ≈ US$0.102/h (us-east-1) + 30 GB gp3 ≈ US$0.003/h. Hard cap 3 h → under US$0.50.
3. **Create**: key pair (`vora-a1-<date>`), security group allowing TCP 22 only from the operator's current IP /32,
   latest Debian 12 / Ubuntu 24.04 arm64 AMI from the public SSM parameter, `InstanceInitiatedShutdownBehavior=terminate`,
   root volume `DeleteOnTermination=true`, user-data `shutdown -h +180` (the box terminates itself after 3 h whatever happens),
   tag `project=vora-a1`.
4. **Evidence**: `lscpu` (expect "Cortex-A72", part 0xd08) saved to results.
5. **Run**: copy the repo (git archive), `docker build -f docker/Dockerfile -t vora:arm64 .` on the instance (native arm64),
   `docker run -d -p 127.0.0.1:8000:8000 vora:arm64`, wait for `/health` 200 (record the time), `scripts/ws_smoke.py`
   with an English and a Chinese sample, then `docker run --rm -v $PWD/results-pi:/app/results vora:arm64 bash scripts/pi_bench.sh`.
6. **Collect**: scp `results-pi/*` back; summarise into `results/deploy.json` → `pi_class_measured`.
7. **Tear down and prove it**: terminate the instance, delete the key pair and the security group; `DescribeInstances`
   (tag project=vora-a1, not terminated) and `DescribeVolumes` / `DescribeKeyPairs` / `DescribeSecurityGroups` must come back
   empty. Report the instance-hours used.

Scaling note: same core as a Pi 4 at 2.3 GHz instead of 1.5 GHz, more memory bandwidth → a Pi 4 should be at least ~1.5×
slower. Reported as an estimate, never as a Pi measurement.
