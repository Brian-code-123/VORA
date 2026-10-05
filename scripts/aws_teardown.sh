#!/usr/bin/env bash
# Run inside AWS CloudShell. Removes VORA resources and proves nothing is left.
#   bash scripts/aws_teardown.sh a1    # only the Pi-class test box + its security group (the demo stays)
#   bash scripts/aws_teardown.sh all   # everything tagged project=vora, both security groups and the bucket
set -uo pipefail
WHAT="${1:?usage: aws_teardown.sh a1|all}"
export AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"
names=$([ "$WHAT" = all ] && echo "vora-demo,vora-a1" || echo "vora-a1")
ids=$(aws ec2 describe-instances --filters Name=tag:project,Values=vora Name=tag:Name,Values="$names" \
      Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$ids" ]; then aws ec2 terminate-instances --instance-ids $ids >/dev/null; aws ec2 wait instance-terminated --instance-ids $ids; fi
for g in $([ "$WHAT" = all ] && echo "vora-a1-sg vora-demo-sg" || echo "vora-a1-sg"); do
  gid=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$g" --query 'SecurityGroups[0].GroupId' --output text)
  [ "$gid" != "None" ] && aws ec2 delete-security-group --group-id "$gid"
done
if [ "$WHAT" = all ]; then
  B="vora-deploy-$(aws sts get-caller-identity --query Account --output text)"
  aws s3 rb "s3://$B" --force >/dev/null 2>&1 || true
fi
echo "--- left over (must be empty) ---"
aws ec2 describe-instances --filters Name=tag:project,Values=vora Name=tag:Name,Values="$names" \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text
aws ec2 describe-volumes --filters Name=tag:project,Values=vora Name=status,Values=available,in-use --query 'Volumes[].VolumeId' --output text | \
  { [ "$WHAT" = all ] && cat || true; }
for g in $([ "$WHAT" = all ] && echo "vora-a1-sg vora-demo-sg" || echo "vora-a1-sg"); do
  aws ec2 describe-security-groups --filters Name=group-name,Values="$g" --query 'SecurityGroups[].GroupId' --output text
done
echo "--- end ---"
