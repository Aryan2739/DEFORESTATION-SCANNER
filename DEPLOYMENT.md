# Production Deployment Guide: Canopy Loss Observatory

This repository is prepared for a Docker-based public deployment. The app downloads Sentinel-2 and Hansen reference rasters at runtime, so select a host with enough memory and check its current plan limits before launching.

## 1. Locked Python dependencies (`requirements.txt`)

The application dependencies are pinned to exact versions in [`requirements.txt`](./requirements.txt). The pins include NumPy 2.5.4, SciPy 1.18.1, and Rasterio 1.5.2, which require Python 3.12 or newer. The Docker image uses Python 3.12. For Streamlit Community Cloud, select Python 3.12 in the deployment's Advanced settings. Use the same Python version locally and install with `pip install -r requirements.txt`.

## 2. Linux native libraries (`packages.txt`)

[`packages.txt`](./packages.txt) lists `gdal-bin` and `libgdal-dev`, one package per line. Streamlit Community Cloud installs these Debian packages from the repository-root file. The Dockerfile installs them directly with `apt`.

## 3. Choose and configure a host

The included [`Dockerfile`](./Dockerfile) installs the pinned dependencies, runs as a non-root user, listens on the host-provided `$PORT` (7860 by default), and exposes Streamlit's `/_stcore/health` endpoint.

- **Render or Railway:** create a Docker web service and choose a plan with at least 2 GB RAM; 4 GB gives more room for multiple visitors. Set the health check to `/_stcore/health` where the host provides that option. Confirm the service's current memory allocation and pricing before launch.
- **Hugging Face Spaces:** CPU Basic provides 2 vCPU and 16 GB RAM with no hourly compute charge. However, running a Docker Space on CPU Basic requires a PRO subscription for an individual account (or Team/Enterprise for an organization), and Spaces can sleep when idle on free hardware. Hugging Face custom domains also require PRO or Team/Enterprise. So CPU Basic is not a no-subscription free deployment option for this Docker app.
- **Streamlit Community Cloud:** it automatically reads the root `requirements.txt` and `packages.txt`, but its resource limits are constrained and can vary. It is a reasonable smoke-test target; for public multi-user raster processing, use a host with explicit RAM allocation.

For a first launch, deploy one container and verify a small sample analysis before increasing traffic or instance count. The app limits the maximum analysis area, caches composites, and allows only one expensive analysis at a time per process. This reduces concurrent memory peaks, but it is not a per-user rate limit; a Cloudflare rule alone cannot provide that compute guard or stop sequential abuse (see the next section).

## 4. Custom domain and Cloudflare

1. Deploy the app and verify its provider URL first.
2. Add the custom hostname in the hosting provider's settings, if required. Prefer a subdomain such as `canopy.example.com`; use the CNAME target supplied by the host. For Hugging Face, add the hostname in the Space's Custom Domain setting and point its CNAME to `hf.space` as instructed there. This feature requires PRO or Team/Enterprise.
3. Add the matching CNAME in Cloudflare DNS. Turn on proxying only after the host confirms the custom hostname and certificate are ready. Cloudflare flattens proxied CNAMEs, but the host still has to accept the hostname. For a root/apex domain, follow the host's own custom-domain instructions rather than assuming every host accepts `@` as a CNAME.
4. In Cloudflare, use **Full (strict)** after the origin has a valid certificate, enable **Always Use HTTPS**, and make sure **Network → WebSockets** is on. Test a full analysis through the custom hostname; Streamlit uses a long-lived WebSocket connection.
5. Optional response headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, and `Referrer-Policy: strict-origin-when-cross-origin`. Check the app's map and embedded image components before applying a restrictive Content Security Policy.

Cloudflare adds an edge proxy and DDoS protections, but its HTTP rate-limit rules only see the initial WebSocket handshake after the connection is established. Streamlit sends analysis actions inside that connection, so a rule such as “30 requests per 10 seconds” on the whole hostname does not reliably limit STAC jobs and can interfere with visitors. Use an application-side or origin-side concurrency/rate limit if the app will accept untrusted public traffic. Cloudflare's free plan currently allows one rate-limit rule with a 10-second counting period; the host's feature availability can change, so confirm it in the dashboard.

## 5. Production Streamlit settings

The project config at [`.streamlit/config.toml`](./.streamlit/config.toml) hides error details and minimizes the toolbar. It uses the current `showErrorDetails = "none"` value; Streamlit documents `false` as deprecated and maps it to stacktrace details. Satellite and reference-data failures are logged server-side and shown to visitors as short, actionable messages.

The config also keeps XSRF protection enabled. If the app is placed behind another proxy, test map interactions, downloads, and WebSocket reconnects after enabling it.

## Launch checklist

- Push this repository to the selected host and confirm the health endpoint reports healthy.
- Run a small analysis through the provider URL, then through the custom hostname after DNS and TLS are ready.
- Check host logs and memory during that run. Keep the app at one instance until session affinity and shared cache behavior are understood.
- Add an application-side per-user rate limit or authentication before inviting substantial untrusted traffic; the process-wide lock only prevents concurrent runs.