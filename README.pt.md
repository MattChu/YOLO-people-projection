# YOLO-people-projection

*[English](README.md)*

Seguir pessoas a andar sobre um chão com uma câmara, saber onde cada uma está,
em metros reais, e projetar imagens no chão exatamente debaixo dos seus pés.

Uma câmara olha de cima para um retângulo marcado com fita no chão. O
[YOLO](https://docs.ultralytics.com/) encontra cada pessoa e os seus
tornozelos, o seguidor (*tracker*) acompanha-as de imagem para imagem, e as
suas posições são enviadas por [OSC](https://opensoundcontrol.stanford.edu/)
para o que quiser: os sketches de [Processing](https://processing.org/)
incluídos, TouchDesigner, Max, Pure Data, o seu próprio Python, qualquer coisa
que fale OSC. Um projetor calibrado para o mesmo retângulo permite a um sketch
desenhar em metros e fazê-lo aparecer no chão onde as pessoas estão.

**[O guia ilustrado](https://mattchu.github.io/YOLO-people-projection/pt/)**
(também em [docs/pt/index.html](docs/pt/index.html)) mostra a montagem física,
a calibração e como as peças encaixam umas nas outras, com diagramas. Imprima-o
para PDF a partir do navegador se quiser uma cópia em papel.

> **Testado em:** Ubuntu 26.04 (Linux), um MacBook Pro de 2017, uma câmara a
> 640×480 (a imagem a cores de uma Intel RealSense D435, usada como câmara
> normal) e um projetor de 1280×800. Os passos para Windows e macOS foram
> escritos a partir da documentação de cada ferramenta, mas ainda não foram
> experimentados; abra uma *issue* se algo não funcionar.

Os comandos, nomes de ficheiros e opções ficam em inglês, tal como se
escrevem.

---

## Índice

1. [O que é preciso](#1-o-que-é-preciso)
2. [Instalar o software](#2-instalar-o-software)
3. [Preparar o espaço](#3-preparar-o-espaço)
4. [Calibrar](#4-calibrar)
5. [Pôr a funcionar](#5-pôr-a-funcionar)
6. [As mensagens OSC, e como usá-las noutros programas](#6-as-mensagens-osc-e-como-usá-las-noutros-programas)
7. [Fazer o seu próprio sketch de Processing](#7-fazer-o-seu-próprio-sketch-de-processing)
8. [Os ficheiros](#8-os-ficheiros)
9. [Resolução de problemas](#9-resolução-de-problemas)

---

## 1. O que é preciso

**Material**

- **Um computador** com Linux, Windows ou macOS. Qualquer portátil recente
  serve; uma placa gráfica NVIDIA torna o seguimento muito mais rápido, mas não
  é necessária.
- **Um projetor**, ligado como segundo ecrã (a estender o ambiente de
  trabalho, não a duplicá-lo), apontado para o chão.
- **Uma câmara** que veja toda a área projetada: serve qualquer câmara que o
  computador reconheça como câmara, desde uma webcam a uma câmara USB ou de
  placa de captura de gama alta, ou a imagem a cores de uma câmara de
  profundidade como a Intel RealSense (que aparece como uma câmara normal).
  Qualquer resolução e cadência de imagens servem: uma câmara mais rápida segue
  as pessoas com mais suavidade se o computador acompanhar, e uma imagem maior
  ajuda a encontrar pessoas que nela aparecem pequenas. Monte-a **alta, a olhar
  para baixo**. Quanto mais a direito olhar para o chão, melhor encontra os pés
  das pessoas.
- **Fita-cola de papel e uma fita métrica**, para marcar um retângulo no chão.

**Software**

- **Python 3.14** (outras versões recentes podem funcionar, mas as versões
  fixadas foram testadas no 3.14).
- **git**, para descarregar este repositório.
- **Processing 4** (testado com o 4.5.7), para os sketches incluídos. Não é
  preciso se só for usar os dados OSC noutro software.

---

## 2. Instalar o software

### 2.1 Obter o Python, o git e o Processing

| | Python 3.14 | git | Processing 4 |
|---|---|---|---|
| **Linux** (Ubuntu 26.04) | `sudo apt install python3 python3-venv` | `sudo apt install git` | [processing.org/download](https://processing.org/download) |
| **Windows** | [python.org/downloads](https://www.python.org/downloads/) (marque "Add python.exe to PATH") | [git-scm.com](https://git-scm.com/download/win) | [processing.org/download](https://processing.org/download) |
| **macOS** | [python.org/downloads](https://www.python.org/downloads/) | `xcode-select --install` | [processing.org/download](https://processing.org/download) |

### 2.2 Descarregar este repositório

```sh
git clone https://github.com/MattChu/YOLO-people-projection.git
cd YOLO-people-projection
```

Todos os comandos a seguir correm dentro desta pasta.

### 2.3 Criar um ambiente Python

Os pacotes Python do projeto ficam numa pasta própria, `yolo-env/`, para não
entrarem em conflito com mais nada no seu computador.

```sh
# Linux e macOS
python3 -m venv yolo-env

# Windows (PowerShell)
py -3.14 -m venv yolo-env
```

### 2.4 Instalar os pacotes

Escolha a linha para o seu computador. Os ficheiros `requirements-…lock`
fixam cada pacote num ficheiro exato e verificado; o `requirements.txt` lista
só os principais, para sistemas que ainda não têm um ficheiro *lock*.

**Linux, sem placa gráfica NVIDIA** (há um *lock* para este caso):

```sh
yolo-env/bin/pip install --require-hashes --only-binary :all: \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-linux-cpu.lock
```

**Windows, sem placa gráfica NVIDIA:**

```powershell
yolo-env\Scripts\pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
yolo-env\Scripts\pip install -r requirements.txt
```

**Linux ou Windows com placa gráfica NVIDIA:** instale primeiro a versão CUDA
do PyTorch. Corra `nvidia-smi` para ver que versão de CUDA o seu controlador
suporta: o PyTorch 2.14.1 existe como `cu130` (CUDA 13.0, controladores
recentes) ou `cu126` (CUDA 12.6, mais antigos).

```sh
yolo-env/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
yolo-env/bin/pip install -r requirements.txt
```

(no Windows, `yolo-env\Scripts\pip` em vez de `yolo-env/bin/pip`)

**macOS:**

```sh
yolo-env/bin/pip install -r requirements.txt
```

**Opcional: fixar o que instalou.** Num sistema sem *lock*, pode criar um,
para que da próxima vez se instalem exatamente os mesmos ficheiros:

```sh
yolo-env/bin/python lock_requirements.py               # sem NVIDIA: p. ex. requirements-windows-cpu.lock
yolo-env/bin/python lock_requirements.py --torch cu130 # NVIDIA, CUDA 13.0
yolo-env/bin/python lock_requirements.py --torch pypi  # macOS
```

### 2.5 Descobrir o número da sua câmara

As câmaras escolhem-se por número: a 0 é normalmente a do próprio portátil, e
as outras contam a partir daí. Com a câmara ligada (e o ambiente ativo, ver
2.6):

```sh
python cameras.py              # lista as câmaras que encontra
python cameras.py --snapshots  # e guarda uma imagem de cada: camera_0.jpg, camera_1.jpg...
```

Num portátil com uma câmara USB ligada, mostra algo como:

```
  camera 0: 1280x720 (V4L2), average brightness 104
  camera 2: 1920x1080 (V4L2), average brightness 97
Use one with --source N. ...
```

Aqui, a câmara 0 é a do próprio portátil e a câmara 2 é a câmara USB, por isso
usaria `--source 2`. A resolução é a que a câmara dá por omissão (pode pedir
outras, ver 4.1); entre parênteses está a forma como o sistema comunica com
ela (V4L2 no Linux, MSMF ou DSHOW no Windows, AVFOUNDATION no macOS). Os
números podem saltar: alguns dispositivos não são câmaras. Se não tiver a
certeza de qual é qual, veja as imagens `camera_N.jpg`. Uma câmara com várias
imagens (a profundidade, o infravermelho e a cor de uma câmara de
profundidade, por exemplo) aparece com vários números; escolha a que dá uma
imagem a cores normal.

**Uma câmara por HDMI** (uma DSLR, uma câmara mirrorless ou uma câmara de
vídeo) funciona através de um dispositivo de captura HDMI para USB, que
aparece aqui como uma câmara qualquer. Ative a saída HDMI limpa (*clean
HDMI*) da câmara, para que a informação do ecrã dela não apareça na imagem, e
desative o desligar automático.

Da primeira vez, o sistema pode pedir autorização para usar a câmara: no
macOS, autorize o terminal em **Definições do Sistema → Privacidade e
Segurança → Câmara**; no Windows, ative **Definições → Privacidade e segurança
→ Câmara → Permitir que as aplicações de ambiente de trabalho acedam à
câmara**.

### 2.6 Usar o ambiente

Sempre que abrir um terminal para correr os scripts, mude primeiro para o
ambiente do projeto:

```sh
source yolo-env/bin/activate        # Linux e macOS
yolo-env\Scripts\Activate.ps1       # Windows PowerShell
```

A linha de comandos passa a começar por `(yolo-env)`, e `python` passa a ser o
Python do projeto. (No Windows, se o PowerShell se recusar a correr o script de
ativação, corra uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)
Em vez de ativar, também pode correr diretamente o Python do ambiente, p. ex.
`yolo-env/bin/python track_osc.py`. O VS Code faz isto por si depois de
escolher `yolo-env` em **Python: Select Interpreter**.

Confirme que está tudo instalado correndo os testes:

```sh
python test_osc.py
python test_calibration.py
python test_follow.py
python test_foot_calibrate.py
python test_recording.py
```

O modelo YOLO (`yolo26n-pose.pt`, cerca de 8 MB) descarrega-se sozinho da
primeira vez que um script precisa dele.

---

## 3. Preparar o espaço

O [guia](https://mattchu.github.io/YOLO-people-projection/pt/#setup) tem
diagramas de tudo isto.

1. **Aponte o projetor** para o chão. Não precisa de ficar de frente: a
   calibração corrige qualquer ângulo.
2. **Ligue-o como segundo ecrã**, a estender o ambiente de trabalho. Tome nota
   da sua **resolução** e da sua **posição** no ambiente de trabalho, em
   píxeis, nas definições de ecrã. Por exemplo, um projetor de 1280×800 colocado
   à direita de um ecrã de portátil com 1920 píxeis de largura está em
   `1920,0`. No Linux, `xrandr --listmonitors` mostra as duas coisas, p. ex.
   `1280/339x800/212+2560+0` é 1280×800 em `2560,0`. No Windows, ponha a escala
   do ecrã do projetor a 100%, para que os seus píxeis sejam píxeis reais.
3. **Monte a câmara** alta, a olhar para baixo, de modo a ver toda a área
   projetada com alguma folga.
4. **Marque um retângulo** no chão com fita-cola, **dentro** da área projetada
   e à vista da câmara: chegam quatro pequenas marcas nos cantos. **Meça a sua
   largura e altura em metros.**
5. **Numere os cantos de cabeça:** a marca 1 é a origem (0, 0), a marca 2 fica
   ao longo da largura (W, 0), a marca 3 é o canto oposto (W, H) e a marca 4 é
   (0, H). Tudo (as posições OSC, os sketches) se mede em metros a partir da
   marca 1, com x ao longo da largura e y ao longo da altura.

---

## 4. Calibrar

A calibração descobre como a câmara, o projetor e o chão se alinham, e
guarda-o em `calibration.json`. **Repita-a sempre que a câmara ou o projetor
mudarem de sítio.** Calibre com a sala **o mais escura possível**.

### 4.1 `calibrate.py`: câmara, projetor e chão

```sh
python calibrate.py --source 2 --proj-pos 1920,0 --proj-size 1280x800 --floor-size 2.4x1.4
```

Use o número da sua câmara (2.5), a posição e a resolução do seu projetor, e a
largura × altura do seu retângulo em metros. A câmara dá a sua própria
resolução e cadência, a não ser que peça outras com `--width`, `--height` e
`--fps`; muitas câmaras USB só dão as maiores resoluções ou as cadências mais
altas com imagens comprimidas, por isso acrescente `--fourcc MJPG` nesses
casos. Outros exemplos:

```sh
# uma câmara USB a 1920x1080 e 60 imagens por segundo, com imagens comprimidas
python calibrate.py --source 2 --width 1920 --height 1080 --fps 60 --fourcc MJPG \
    --proj-pos 1920,0 --proj-size 1280x800 --floor-size 2.4x1.4

# uma câmara por HDMI, através do dispositivo de captura 1 (a exposição acerta-se na própria câmara)
python calibrate.py --source 1 --width 1920 --height 1080 --fourcc MJPG \
    --proj-pos 1920,0 --proj-size 1280x800 --floor-size 2.4x1.4

# os marcadores parecem queimados: fixar a exposição (unidades do Linux; ver abaixo)
python calibrate.py --source 2 --exposure 80 --proj-pos 1920,0 --proj-size 1280x800 --floor-size 2.4x1.4
```

O número da câmara, a resolução, e `--fps` e `--fourcc` se os usou, ficam
guardados com a calibração, por isso os scripts seguintes (`foot_calibrate.py`,
`follow.py`, `track_osc.py`) usam a mesma câmara da mesma maneira sem ser
preciso dizer-lhes: não é preciso `--source`. Só `--exposure` não fica
guardado: volte a dá-lo se o usou, ou acerte a exposição de uma vez na própria
câmara.

Depois, com as teclas na janela da câmara:

1. **Marcadores.** O projetor mostra uma grelha de marcadores quadrados.
   Mantenha o chão livre e carregue em **ESPAÇO**. São encontrados ao longo de
   algumas imagens, e dão a correspondência entre a câmara e o projetor. Se
   forem encontrados poucos, veja a [Resolução de problemas](#9-resolução-de-problemas).
2. **Chão.** O projetor fica branco. **Clique nas quatro marcas de fita** na
   imagem da câmara, pela ordem 1, 2, 3, 4 (**u** desfaz um clique).
3. **Verificação.** O projetor desenha uma grelha em coordenadas do chão. O
   seu contorno deve ficar em cima da fita. **s** guarda, **r** recomeça, **q**
   sai sem guardar.

**Se os marcadores parecerem queimados** (manchas brancas em vez de quadrados
nítidos a preto e branco), fixe a exposição da câmara: numa sala escura, a
exposição automática aclara a imagem por causa da sala e sobre-expõe a
projeção, que é muito luminosa. A opção `--exposure` usa as unidades da
própria câmara, que dependem do sistema: no Linux, normalmente décimas de
milissegundo (experimente `--exposure 80`, depois 40 ou 160); no Windows, uma
potência de 2 em segundos (experimente `--exposure -6`, depois -7 ou -5). Nem
todas as câmaras, nem todos os sistemas, deixam os programas mudá-la.

### 4.2 `foot_calibrate.py`: onde assentam os pés

O YOLO vê tornozelos, que ficam um pouco acima e atrás do ponto onde o pé toca
no chão. Este passo mede essa diferença e corrige-a.

```sh
python foot_calibrate.py
```

O projetor mostra alvos numerados, um de cada vez. **Ponha-se com a cruz entre
os pés** e fique quieto: passado um momento, o alvo é registado e acende-se o
seguinte, por isso uma pessoa consegue fazê-lo sozinha. No fim, **s** guarda.
(ESPAÇO regista já, **n** salta um alvo, **b** volta ao anterior.)

### 4.3 `follow.py`: confirmar que funciona

```sh
python follow.py
```

É projetado um círculo debaixo dos pés de cada pessoa. Ande pelo espaço: o
círculo deve segui-lo. **g** mostra ou esconde uma grelha no chão (começa escondida), **q** sai.

---

## 5. Pôr a funcionar

Num terminal, arranque o seguidor. Ele envia a posição de toda a gente por
OSC:

```sh
python track_osc.py --show        # --show abre também uma janela da câmara, para confirmar
```

Depois corra um sketch. Abra um destes no Processing e carregue em Run:

- `processing/template/template.pde`: um círculo nos pés de cada pessoa. O
  ponto de partida para os seus.
- `processing/grid_peeps/grid_peeps.pde`: um chão cheio de logótipos que se
  afastam à volta das pessoas. Carregue em **m** para os outros modos: linhas
  que passam por cada pessoa, com logótipos a rodopiar entre elas, e um rasto
  de logótipos atrás de cada pessoa.
- `processing/pong/pong.pde`: duas pessoas jogam pong ao longo do chão.

O sketch abre em ecrã inteiro no projetor, se estiver ligado, ou numa janela
de teste, se não estiver.

**Sem câmara** (ou sem pessoas), há pessoas imaginárias a passear pelo chão:

```sh
python track_osc.py --simulate 3                     # três pessoas imaginárias, sem câmara
python track_osc.py --simulate 2 --with-camera --show # você, e duas pessoas imaginárias
```

Nos sketches, manter o botão do rato carregado também cria uma pessoa de teste
no sítio do rato.

**Opções úteis** (`python track_osc.py --help` lista-as todas):

| Opção | O que faz |
|---|---|
| `--show` | janela da câmara com as deteções e o seguimento desenhados por cima |
| `--osc-host 192.168.1.20` | enviar para outro computador (por omissão: este) |
| `--osc-port 12000` | a porta OSC (por omissão 12000) |
| `--max-fps 15` | seguir no máximo este número de imagens por segundo (por omissão: todas as que a câmara dá); menos deixa mais processador livre para um sketch no mesmo computador |
| `--imgsz 320` | o tamanho de imagem do YOLO; mais pequeno é mais rápido, maior encontra pessoas mais pequenas |
| `--device cpu` / `--device 0` | correr o YOLO no processador, ou na primeira placa NVIDIA (por omissão: a placa, se houver) |
| `--threads 1` | *threads* do processador para o YOLO; mais quase não ajuda num portátil e tira processador ao sketch |
| `--smooth 0.5` | suavização das posições; 0 = em bruto, mais perto de 1 = mais suave mas com mais atraso |
| `--dark-objects` | seguir objetos escuros em vez de pessoas (para testar numa mesa) |

**Num portátil**, o seguidor e o sketch disputam o processador. Se o sketch
der solavancos, experimente `--max-fps 15` (ou menos) ou `--imgsz 256`. Com uma
placa gráfica NVIDIA, o seguidor consegue normalmente acompanhar todas as
imagens de uma câmara rápida.

---

## 6. As mensagens OSC, e como usá-las noutros programas

O `track_osc.py` envia uma mensagem UDP por cada pessoa seguida, em cada
imagem que segue (tantas vezes quanto a câmara e o computador permitirem, ou
`--max-fps`), por omissão para `127.0.0.1:12000`:

```
/person  id x y
```

| Argumento | Tipo | Significado |
|---|---|---|
| `id` | int | o mesmo enquanto a pessoa estiver a ser seguida |
| `x` | float | metros a partir da marca 1, ao longo da largura (0 a W) |
| `y` | float | metros a partir da marca 1, ao longo da altura (0 a H) |

**Não se envia nada quando alguém sai.** Esqueça quem não tiver dado notícias
há quase um segundo (isso também cobre pacotes perdidos). Se o seguidor
perder alguém, guarda o seu id durante uns segundos (`--reattach-time`): se
voltar a encontrá-lo perto de onde o perdeu
(`--reattach-dist`), envia-o outra vez com o mesmo id. Por isso, guarde durante
uns segundos quem deixou de dar notícias, se quiser que volte como a mesma
pessoa.

Recebê-las noutros programas:

- **Processing:** a aba `People` em `processing/template/` (não é preciso
  nenhuma biblioteca). Ver a [secção 7](#7-fazer-o-seu-próprio-sketch-de-processing).
- **TouchDesigner:** um **OSC In DAT** (ou **OSC In CHOP**) na porta de rede
  12000.
- **Max:** `[udpreceive 12000]` → `[route /person]`.
- **Pure Data:** `[netreceive -u -b 12000]` → `[oscparse]` →
  `[list trim]` → `[route person]`.
- **Python**, com `pip install python-osc`:

  ```python
  from pythonosc.dispatcher import Dispatcher
  from pythonosc.osc_server import BlockingOSCUDPServer

  def person(address, pid, x, y):
      print(f"pessoa {pid} em {x:.2f}, {y:.2f} m")

  d = Dispatcher()
  d.map("/person", person)
  BlockingOSCUDPServer(("127.0.0.1", 12000), d).serve_forever()
  ```

Para enviar para software noutro computador, use `--osc-host` com o endereço
IP desse computador, e confirme que a firewall dele deixa entrar a porta UDP
12000.

---

## 7. Fazer o seu próprio sketch de Processing

1. Copie `processing/template` para `my-sketches/<nome>/` (a pasta
   `my-sketches/` é ignorada por este repositório, por isso os seus sketches
   continuam seus: pode até ser um repositório git próprio), e mude o nome de
   `template.pde` para `<nome>.pde`.
2. Desenhe em `canvas`, que é o retângulo da fita a `PIXELS_PER_METRE` (500)
   píxeis por metro, com a origem na marca 1. A aba Projection deforma-o para
   o pôr em cima do chão.
3. `people` tem toda a gente que está no chão: `p.x`, `p.y` em metros (ou
   `p.canvasX()`, `p.canvasY()` em píxeis do canvas), suavizados entre
   mensagens; `p.shown` vai de 0 a 1 quando a pessoa chega e volta a 0 quando
   sai; `p.id` é o id do seguidor, que se mantém durante falhas curtas.
4. `personArrived(p)` e `personLeft(p)`, na aba principal, são chamadas quando
   as pessoas chegam e saem.

Mantenha as abas `People` e `Projection` iguais em todos os sketches (copie-as
para os outros se as alterar). Um sketch encontra o `calibration.json` na sua
própria pasta ou na mais próxima acima dela, por isso funciona em qualquer
sítio dentro desta pasta.

---

## 8. Os ficheiros

| Ficheiro | O que é |
|---|---|
| `calibrate.py` | passo 4.1: calibração câmara ↔ projetor ↔ chão |
| `foot_calibrate.py` | passo 4.2: correção do sítio onde assentam os pés |
| `follow.py` | passo 4.3: projeta um círculo debaixo de cada pessoa, para confirmar o seguimento |
| `track_osc.py` | o seguidor para pôr uma peça a funcionar: envia OSC, sem janela de projetor |
| `p_track.py` | um teste rápido do YOLO numa câmara ou vídeo, sem precisar de calibração |
| `pipeline.py` | imagem da câmara → pés das pessoas → metros no chão → seguimento (partilhado por follow e track_osc) |
| `models.py` | carrega o YOLO para o hardware desta máquina (`--device`, `--threads`) |
| `tracking.py` | acompanha as posições de imagem para imagem, suavizando-as e mantendo os ids |
| `cameras.py` | imagens de qualquer câmara ou de um ficheiro de vídeo; corra-o para listar as suas câmaras |
| `calib_utils.py` | a matemática da calibração, os marcadores e funções para as janelas |
| `osc.py` | envia as mensagens OSC (não é preciso nenhum pacote extra) |
| `recording.py` | grava a imagem da câmara, do projetor ou do ecrã (teclas c / p / f) |
| `calibration.json` | criado pela calibração: as medidas do seu espaço (não está no repositório) |
| `processing/` | os sketches de Processing |
| `requirements.txt`, `requirements-*.lock`, `lock_requirements.py` | os pacotes Python |
| `test_*.py` | testes do código de calibração, seguimento, OSC e gravação |

A forma como se ligam está desenhada no [guia](https://mattchu.github.io/YOLO-people-projection/pt/#files).

---

## 9. Resolução de problemas

**A calibração encontra poucos marcadores.**
- Escureça mais a sala.
- Fixe a exposição, ou experimente outra: ver o fim de 4.1.
- Use marcadores maiores: `--grid 6x4` ou `--grid 5x3`.
- Aponte a câmara mais para baixo, para que a projeção ocupe mais da imagem.
  Marcadores vistos de um ângulo muito rasante ficam demasiado achatados para
  se lerem.

**Os pés são seguidos no sítio errado.** Volte a correr o
`foot_calibrate.py`. Se a câmara ou o projetor mudaram de sítio, refaça
primeiro o `calibrate.py`.

**O sketch dá solavancos enquanto o seguidor corre.** Baixe a carga do
seguidor: `--max-fps 15`, `--imgsz 256`, mantenha `--threads 1`, ou corra o
seguidor num segundo computador com `--osc-host`.

**Ninguém é detetado.** As pessoas podem estar demasiado pequenas na imagem:
aproxime a câmara do chão, ou experimente `--imgsz 480`.

**Uma janela abre no ecrã errado, ou o ecrã inteiro porta-se mal** (no macOS,
sobretudo). Use `--windowed` no `calibrate.py`, `foot_calibrate.py` e
`follow.py` para pôr a imagem do projetor numa janela normal, e depois
arraste-a para o projetor.

**Linux: os sketches de Processing falham com `UnsatisfiedLinkError`** (todos
os sketches P2D/P3D, até os exemplos). Feche o Processing e acrescente esta
linha a `~/.config/processing/preferences.txt`:

```
run.options=-Djdk.lang.Process.launchMechanism=VFORK
```

**Linux: um sketch que usa a biblioteca Sound morre com "Target VM failed to
initialize"** (o PipeWire anuncia 128 canais de áudio). Crie `~/.asoundrc` com
o dispositivo do PipeWire por omissão, limitado a 2 canais:

```
pcm.!default {
    type pipewire
    playback_node "-1"
    capture_node  "-1"
    channels 2
}
```

**Linux: as janelas parecem congeladas depois de fechar uma janela de ecrã
inteiro.** Ligue o compositor do ambiente de trabalho (XFCE: Window Manager
Tweaks → Compositor).

**Linux: a gravação do ecrã (tecla f) não funciona.** Precisa de uma sessão
X11, não Wayland.

**A câmara desaparece ou não abre.** Confirme que mais nenhum programa a está
a usar (só um de cada vez pode usá-la), e volte a correr `python cameras.py`:
os números podem mudar quando as câmaras são ligadas por outra ordem. Uma
câmara de alta resolução pode precisar de uma porta e de um cabo USB 3. Alguns
portáteis perdem o controlador USB 3 depois de suspender; desligar por
completo, e nos Mac fazer um reset ao SMC, resolve.

---

## Licença

[MIT](LICENSE).
