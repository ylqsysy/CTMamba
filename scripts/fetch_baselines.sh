#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dst="$root/external/baselines"
lock="$root/baselines/BASELINES.lock.txt"

mkdir -p "$dst"
mkdir -p "$root/baselines"

# 如果你已经在 WSL 里设置了 git --global proxy，这里不强制再写一次。
# 但为了脚本稳定，每次 clone 都用 -c 强制 HTTP/1.1 + 更长 low-speed 窗口。
git_clone_retry () {
  local url="$1"
  local out="$2"
  local tries=3
  local i=1

  while [ $i -le $tries ]; do
    echo "[clone] attempt $i/$tries -> $out"
    if git \
      -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1 \
      -c http.lowSpeedTime=600 \
      clone --depth 1 --filter=blob:none --no-tags "$url" "$out"; then
      return 0
    fi
    echo "[warn] clone failed: $out"
    rm -rf "$out"
    sleep $((2*i))
    i=$((i+1))
  done

  echo "[error] clone failed after $tries tries: $url"
  return 1
}

# name | repo_url
repos=(
  "spectralformer|https://github.com/danfenghong/IEEE_TGRS_SpectralFormer.git"
  "ssftt|https://github.com/zgr6010/HSI_SSFTT.git"                         # 
  "gsc_vit|https://github.com/flyzzie/TGRS-GSC-VIT.git"                     # :contentReference[oaicite:2]{index=2}
  "morphformer|https://github.com/mhaut/morphFormer.git"              # 
  "mambahsi|https://github.com/RockAilab/MambaHSI_Plus.git"                # 
  "3dss_mamba|https://github.com/IIP-Team/3DSS-Mamba.git"                   # 该仓库是否包含完整训练代码需后续检查
  "a2s2k|https://github.com/suvojit-0x55aa/A2S2K-ResNet.git"               # 
  "igroupss_mamba|https://github.com/IIP-Team/IGroupSS-Mamba.git"          # 
)

echo "# baseline repos pinned by commit" > "$lock"
echo "# generated: $(date -Iseconds)" >> "$lock"

for item in "${repos[@]}"; do
  name="${item%%|*}"
  url="${item#*|}"
  out="$dst/$name"

  if [ -d "$out/.git" ]; then
    echo "[skip] $name exists"
  else
    git_clone_retry "$url" "$out"
  fi

  (cd "$out" && commit="$(git rev-parse HEAD)" && echo "$name $commit $url" >> "$lock")
done

echo "[ok] wrote $lock"
