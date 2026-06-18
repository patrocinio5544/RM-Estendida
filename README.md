# MedNeXt — RM-Estendida — Treino em VM Linux (RunPod)

Guia operacional para treinar o **MedNeXt** sobre o dataset **Task220_PI-CAI**
em uma VM Linux com GPU (referência: **RunPod**), partindo de um pacote
`pi_cai220_mednext.tar.zst` que contém **apenas o dataset raw** (imagens, labels,
`dataset.json`, `splits_final.json`). O pré-processamento e o *integrity check*
rodam na própria VM.

> Branch desta documentação: `RM-Estendida-VM`.
> A rota Docker/Compose continua válida para WSL/local e está no `README.md` original.
> Em RunPod o caminho mais eficiente é rodar **nativo no pod** (a imagem já é um
> container com torch), sem Docker-in-Docker.

---

## 1. Sumário: o que deu certo e o que deu errado (ambiente local / WSL)

Registro das lições da validação local, para não repetir os mesmos tropeços na VM.

**Funcionou:**
- `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` + torch 2.1.0 cu118 é a base estável.
- Docker Engine instalado pelo repositório oficial + NVIDIA Container Toolkit.
- O pacote `.tar.zst` traz a árvore a partir de `data/`, então extrai na **raiz do projeto**.
- `tar -I zstd -tvf` para inspecionar antes de extrair evitou duplicar pastas.

**Tropeços e correções:**
- **`[boot]` / `systemd=true` digitados no terminal** → eram conteúdo do arquivo
  `/etc/wsl.conf`, não comandos. Como o `systemctl` já respondia, o systemd já
  estava ativo e o `wsl.conf` foi dispensável.
- **`permission denied` no `docker.sock`** → faltava aplicar o grupo `docker` na
  sessão. Resolvido com `usermod -aG docker $USER` + `newgrp docker` (ou
  `wsl --shutdown` e reabrir).
- **`TLS handshake timeout` ao puxar imagem** → conectividade/DNS do WSL.
  Mitigado fixando DNS (`generateResolvConf = false` + `resolv.conf` com 1.1.1.1/8.8.8.8).
- **Expectativa de "treinar direto"** → o pacote **não** tinha `nnUNet_preprocessed`
  nem `plans`. Foi preciso rodar `mednextv1_plan_and_preprocess` (que também faz o
  *integrity check*) antes do treino.
- **Nome da task com hífen** (`Task220_PI-CAI`) → usar sempre o **número** (`220`)
  nos comandos do nnUNet/MedNeXt.

---

## 2. Escolha da GPU por tamanho de modelo

MedNeXt v1 é 3D com patch `128×128×128` e *spacing* 1mm isotrópico — é pesado em VRAM.
Valores abaixo são **estimativas** (variam com batch, deep supervision e gradient
checkpointing). MedNeXt suporta *gradient checkpointing* para caber modelos grandes
em menos memória, trocando memória por compute.

| Modelo (ID) | Params | Kernel | VRAM mínima viável | GPU confortável (RunPod) |
|-------------|--------|--------|--------------------|--------------------------|
| Small (S)   | 5.6–5.9M  | 3 / 5 | ~12–16 GB | RTX 4090 24 GB / A5000 |
| Base (B)    | 10.5–11M  | 3 / 5 | ~16–24 GB | RTX 4090 24 GB / A6000 48 GB |
| **Medium (M)** | 17.6–18.3M | 3 / 5 | ~24–40 GB | **A100 80 GB / H100 / H200 141 GB** |
| Large (L)   | 61.8–63M  | 3 / 5 | ~40–80 GB | H100 80 GB / **H200 141 GB** |

**Neste guia o exemplo usa Medium (M) na variante mais pesada (kernel 5 via UpKern),
em uma H200.** A H200 (141 GB) dá folga grande para o Medium: permite kernel 5
confortável, batch maior e não exige gradient checkpointing.

Regra prática: para *desenvolver/depurar* use a menor GPU que couber (mais barata);
para o *treino final* suba para a GPU "confortável" da linha correspondente.

---

## 3. Provisionar a VM no RunPod

1. **Storage primeiro (recomendado):** crie um **Network Volume** (Secure Cloud).
   Ele persiste após o pod ser terminado e monta em `/workspace`. Dimensione para
   o dataset + pré-processado + checkpoints (ex.: 100–200 GB). O volume fica preso
   ao datacenter escolhido, então selecione um datacenter com a GPU desejada.
2. **Deploy do Pod:** em *Pods → Deploy*, anexe o Network Volume **durante a criação**
   (não dá para anexar depois), escolha a GPU (ex.: **H200** para o Medium) e o template.
3. **Template:** use uma imagem PyTorch que já bata com o stack do projeto, p.ex.
   `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`. Assim torch 2.1.0 + CUDA 11.8 já
   vêm prontos e você só instala o MedNeXt por cima.
4. **Acesso:** suba sua chave SSH em *Settings* e conecte via SSH, ou use o terminal web.
5. **Auto-shutdown:** ative timeout de inatividade para não pagar pod ocioso.

> Tudo que você quer preservar entre sessões deve ficar em `/workspace`
> (o Network Volume). O disco do container some quando o pod é terminado.

---

## 4. Preparar o ambiente no pod

Trabalhe sempre dentro de `/workspace` (volume persistente).

```bash
cd /workspace

# ferramentas base
apt-get update && apt-get install -y git zstd build-essential curl

# clonar a branch de VM
git clone -b RM-Estendida-VM https://github.com/patrocinio5544/RM-Estendida.git mednext
cd mednext

# estrutura de pastas (mesma lógica do compose, agora em /workspace)
mkdir -p data/nnUNet_raw data/nnUNet_preprocessed outputs/nnUNet_results
```

Instalar o MedNeXt. Como o template já tem torch 2.1.0 cu118, **não reinstale torch**:

```bash
pip install -r requirements_docker.txt   # mesmas deps validadas da imagem
pip install -e .
```

Exportar os paths do nnUNet (no pod nativo não há compose para setar env).
Coloque no `~/.bashrc` para persistir entre logins:

```bash
cat >> ~/.bashrc <<'EOF'
export nnUNet_raw_data_base=/workspace/mednext/data/nnUNet_raw
export nnUNet_preprocessed=/workspace/mednext/data/nnUNet_preprocessed
export RESULTS_FOLDER=/workspace/mednext/outputs/nnUNet_results
EOF
source ~/.bashrc
```

Validar GPU e paths (o "exportar paths via python3"):

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python3 -c "import os; [print(k,'=',os.environ.get(k)) for k in ['nnUNet_raw_data_base','nnUNet_preprocessed','RESULTS_FOLDER']]"
```

---

## 5. Obter o dataset

Duas rotas. Use **uma**. Em ambos os casos o alvo é
`/workspace/mednext/pi_cai220_mednext.tar.zst`.

### Rota A — Cópia (do seu local para o pod)

Mais simples se o arquivo já está na sua máquina.

**A.1 `runpodctl` (relay com código único, sem IP):**
```bash
# Na sua máquina local (com runpodctl instalado e logado):
runpodctl send pi_cai220_mednext.tar.zst
# ele imprime um código; no pod:
runpodctl receive <CODIGO>
```

**A.2 `rsync`/`scp` via SSH (bom para arquivos grandes, retomável):**
```bash
# Na sua máquina local (pegue host/porta SSH do pod no painel do RunPod):
rsync -avzP -e "ssh -p <PORTA>" \
  pi_cai220_mednext.tar.zst root@<IP_DO_POD>:/workspace/mednext/
# --inplace permite retomar se cair:
# rsync -avzP --inplace -e "ssh -p <PORTA>" ...
```

### Rota B — rclone a partir do Google Drive

Link fixado do dataset:
`https://drive.google.com/file/d/1zU7dQcDF65Ag2bDZ-5neL4raTte8SawN/view?usp=sharing`
(file ID: `1zU7dQcDF65Ag2bDZ-5neL4raTte8SawN`)

**B.0 Atalho para ESTE link específico (mais rápido):** se o arquivo está
compartilhado como "qualquer pessoa com o link", `gdown` baixa direto pelo ID,
sem configurar OAuth:
```bash
pip install gdown
gdown 1zU7dQcDF65Ag2bDZ-5neL4raTte8SawN -O /workspace/mednext/pi_cai220_mednext.tar.zst
```

Se preferir/precisar do **rclone** (necessário quando o arquivo está no *seu* Drive
e não público), siga abaixo.

**B.1 Instalar rclone no pod:**
```bash
curl https://rclone.org/install.sh | bash
rclone version
```

**B.2 Configurar o remote do Google Drive (fluxo headless).**
O pod não tem navegador, então a autorização é feita na sua máquina local.

No **pod**:
```bash
rclone config
# n) New remote
# name> gdrive
# Storage> drive          (Google Drive)
# client_id>              (Enter — em branco)
# client_secret>          (Enter — em branco)
# scope> 1                (Full access)
# service_account_file>   (Enter — em branco)
# Edit advanced config? > n
# Use web browser to automatically authenticate? > n   <-- N, é headless
```
O rclone vai imprimir um comando como:
```
rclone authorize "drive" "eyJ...base64..."
```
Copie esse comando inteiro e rode **na sua máquina local** (que tem navegador e
rclone instalado). O navegador abre, você loga na conta do Drive e autoriza.
O rclone local imprime um **token** (JSON). Copie esse token e cole de volta no
pod, no prompt `config_token>`. Finalize:
```
# Configure this as a Shared Drive (Team Drive)? > n
# y) Yes this is OK
# q) Quit config
```

**B.3 Validar o remote:**
```bash
rclone listremotes            # deve mostrar "gdrive:"
rclone lsd gdrive:            # lista pastas do seu Drive (sanity check)
```

**B.4 Baixar o dataset.** Se o arquivo está na raiz do *seu* Drive:
```bash
rclone copy "gdrive:pi_cai220_mednext.tar.zst" /workspace/mednext/ -P
```
Se o arquivo veio de um link compartilhado (não está no seu "My Drive"), o
caminho mais simples é abrir o link no navegador e **"Adicionar atalho ao Drive"**
(ou copiar para o seu Drive) antes do `rclone copy` — ou usar o atalho `gdown` do B.0.

---

## 6. Extrair + integrity check (pré-processamento)

A partir de `/workspace/mednext`:

```bash
# integridade do arquivo compactado
zstd -t pi_cai220_mednext.tar.zst

# inspecionar a árvore (deve começar em data/nnUNet_raw/...)
tar -I zstd -tvf pi_cai220_mednext.tar.zst | head

# extrair na RAIZ do projeto (o tar já embute data/)
tar -I zstd -xvf pi_cai220_mednext.tar.zst -C .

# conferir
ls data/nnUNet_raw/nnUNet_raw_data/Task220_PI-CAI/
```

Rodar plan + preprocess + **integrity check** (planner custom 1mm isotrópico, que
gera o plano `nnUNetPlansv2.1_trgSp_1x1x1`). É **CPU-bound**, não usa GPU, e pode
levar de minutos a dezenas de minutos:

```bash
mednextv1_plan_and_preprocess -t 220 --verify_dataset_integrity \
  -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 \
  -pl2d None
```

Isso cria `data/nnUNet_preprocessed/Task220_PI-CAI/` com os `.npz`/`.pkl` e o
`nnUNetPlansv2.1_trgSp_1x1x1_plans_3D.pkl`. Como está em `/workspace`, persiste.
Ajuste paralelismo com `-tf` (threads fullres) e `-tl` (threads lowres) conforme os
vCPUs do pod.

---

## 7. Treinar — exemplo Medium (M) pesado em H200

O **Medium mais pesado** é o kernel 5, treinado via **UpKern**: primeiro treina-se
o kernel 3, depois inicializa-se o kernel 5 a partir dele. Sintaxe:
`mednextv1_train 3d_fullres TRAINER 220 FOLD -p nnUNetPlansv2.1_trgSp_1x1x1`.

**Passo 1 — Medium kernel 3 (base para o UpKern):**
```bash
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 220 0 \
  -p nnUNetPlansv2.1_trgSp_1x1x1
```

**Passo 2 — Medium kernel 5 via UpKern (o "mais pesado"):**
```bash
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 220 0 \
  -p nnUNetPlansv2.1_trgSp_1x1x1 \
  -pretrained_weights /workspace/mednext/outputs/nnUNet_results/nnUNet/3d_fullres/Task220_PI-CAI/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_final_checkpoint.model \
  -resample_weights
```
A flag `-resample_weights` dispara o algoritmo UpKern (interpolação trilinear dos
pesos do kernel 3 para o 5).

**Rodar destacado e com log** (treino longo; não perde se a sessão SSH cair):
```bash
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 220 0 \
  -p nnUNetPlansv2.1_trgSp_1x1x1 \
  > /workspace/mednext/train_M_k3_fold0.log 2>&1 &
tail -f /workspace/mednext/train_M_k3_fold0.log
```

**Folds:** o exemplo usa fold `0`. Para validação cruzada completa, repita com
`1`, `2`, `3`, `4`. Acompanhe a GPU com `watch -n 2 nvidia-smi`.

**Notas de hardware (H200 / Medium):** com 141 GB há folga ampla — kernel 5 cabe
sem gradient checkpointing e dá para aumentar workers do dataloader. Se um dia
treinar o **Large**, mantenha-se em H100/H200 e considere gradient checkpointing.

---

## 8. Checkpoints e persistência

- Resultados em `RESULTS_FOLDER` → `/workspace/mednext/outputs/nnUNet_results`.
- Está no Network Volume, então sobrevive ao término do pod.
- Para arquivar/versionar fora do pod, dá para enviar de volta ao Drive com rclone:
  ```bash
  rclone copy /workspace/mednext/outputs/nnUNet_results gdrive:mednext_results -P
  ```

---

## 9. Referências

- MedNeXt (MIC-DKFZ) e nnUNet v1 (base do pipeline).
- RunPod — Pods, Network Volumes (persistência em `/workspace`), transferência de
  arquivos (`runpodctl send/receive`, `rsync`).
- rclone — backend Google Drive e *remote setup* headless (`rclone authorize`).
