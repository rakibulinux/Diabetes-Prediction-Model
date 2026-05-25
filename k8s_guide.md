Here's a full reference for managing your K8s deployment.

## K8s Management Commands

**Deploy / Update:**

```bash
# Apply all resources (creates/updates)
kubectl apply -f k8s-deploy.yml

# Update image tag (after new CI build)
kubectl set image deployment/diabetes-api -n mlops \
  diabetes-api=ghcr.io/rakibulinux/diabetes-prediction-model/diabetes-api:<new-sha>

# Or edit inline
kubectl edit deployment/diabetes-api -n mlops
```

**Delete:**

```bash
kubectl delete -f k8s-deploy.yml          # everything
kubectl delete deployment/diabetes-api -n mlops  # just deployment
kubectl delete service/diabetes-api-service -n mlops
```

**Scale:**

```bash
kubectl scale deployment/diabetes-api -n mlops --replicas=5
```

**Restart (zero-downtime rolling restart):**

```bash
kubectl rollout restart deployment/diabetes-api -n mlops
```

**Check status:**

```bash
kubectl get all -n mlops
kubectl get hpa -n mlops
kubectl rollout status deployment/diabetes-api -n mlops
kubectl describe pod -n mlops  # if pods are failing
```

**Logs:**

```bash
kubectl logs -n mlops -l app=diabetes-api --tail=50 -f
```

**Port-forward (access locally without LoadBalancer):**

```bash
kubectl port-forward -n mlops service/diabetes-api-service 8080:80
# Now hit http://localhost:8080
```

---

## SSL on Local

Use **mkcert** for trusted local certs + an nginx sidecar (or your app's own SSL):

```bash
# 1. Install mkcert & create cert
brew install mkcert    # or apt/whatever
mkcert -install
mkcert localhost

# 2. Create TLS secret
kubectl create secret tls diabetes-api-tls -n mlops \
  --cert=localhost.pem --key=localhost-key.pem
```

Then apply an **Ingress** (`k8s-ingress.yml`):

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: diabetes-api-ingress
  namespace: mlops
spec:
  ingressClassName: nginx
  tls:
    - hosts: [localhost]
      secretName: diabetes-api-tls
  rules:
    - host: localhost
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: diabetes-api-service
                port:
                  number: 80
```

```bash
# Requires an ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.0/deploy/static/provider/cloud/deploy.yaml
kubectl apply -f k8s-ingress.yml
# Now https://localhost works
```

---

## Production-like Local vs Cloud Server

### Local (Minikube / kind / k3s)

```bash
# Start cluster
minikube start --cpus 4 --memory 4096
minikube addons enable ingress    # enables nginx ingress controller
minikube addons enable metrics-server  # required for HPA

# Deploy
kubectl apply -f k8s-deploy.yml

# The Service type=LoadBalancer won't get a real LB on local.
# Fix: change to NodePort, or use minikube tunnel:
minikube tunnel   # gives LoadBalancer a real local IP

# Access
curl http://$(minikube ip)/health
```

**Recommended local Sevice type** – edit `k8s-deploy.yml` to use `NodePort` for local:

```yaml
spec:
  type: NodePort # instead of LoadBalancer
  ports:
    - port: 80
      targetPort: 8000
      nodePort: 30080
```

Then: `curl http://$(minikube ip):30080`

### Cloud (EKS / AKS / GKE)

```bash
# Create cluster (example with eksctl)
eksctl create cluster --name mlops --region us-east-1 --nodes 2 --node-type t3.medium

# Deploy
kubectl apply -f k8s-deploy.yml

# The LoadBalancer type provisions a real cloud LB automatically
kubectl get svc -n mlops diabetes-api-service
# NAME                    TYPE           CLUSTER-IP      EXTERNAL-IP
# diabetes-api-service    LoadBalancer   10.100.x.x      a1234-5678.elb.amazonaws.com

# Add SSL via cert-manager + Let's Encrypt (minimal setup):
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.0/cert-manager.yaml

# Create Issuer + Ingress with cert-manager annotations
# → auto-provisions real TLS certs. See: https://cert-manager.io/docs/
```

---

## Quick Reference: One-Liner by Scenario

| Goal               | Command                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| **Initial deploy** | `kubectl apply -f k8s-deploy.yml`                                             |
| **Update image**   | `kubectl set image deployment/diabetes-api -n mlops diabetes-api=<new-image>` |
| **Safe restart**   | `kubectl rollout restart deployment/diabetes-api -n mlops`                    |
| **Scale up**       | `kubectl scale deployment/diabetes-api -n mlops --replicas=5`                 |
| **Check rollback** | `kubectl rollout history deployment/diabetes-api -n mlops`                    |
| **Rollback**       | `kubectl rollout undo deployment/diabetes-api -n mlops --to-revision=1`       |
| **Access API**     | `kubectl port-forward -n mlops svc/diabetes-api-service 8080:80`              |
| **Tail logs**      | `kubectl logs -n mlops -l app=diabetes-api --tail=50 -f`                      |
| **Debug pod**      | `kubectl exec -n mlops -it deployment/diabetes-api -- sh`                     |
| **Teardown**       | `kubectl delete -f k8s-deploy.yml`                                            |
