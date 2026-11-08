# img23d — imagem para impressão 3D local

Ferramenta para Windows 11, Python 3.11/3.12 e NVIDIA RTX 50. Gera **geometria sem textura** com Hunyuan3D-2 mini ou TripoSR, prepara o sólido e exporta STL e 3MF em milímetros. Uma única imagem exige que a IA estime os lados ocultos: não é reconstrução dimensional fiel.

## Validação neste computador

Em 08/11/2026: Windows, Python 3.11.18, RTX 5060, PyTorch 2.7.1+cu128, capability 12.0. Ambos os modelos executaram a geração completa com bloqueio de rede, usando `models/code/TripoSR/examples/horse.png`, altura final 80 mm e resolução 128:

| Backend | Geração, incluindo carregamento | Pico alocado / reservado PyTorch | Resultado |
|---|---|---|---|
| Hunyuan mini | 66,8 s | 2250,9 / 2358 MiB | STL/3MF, watertight, 95.108 triângulos |
| TripoSR | 38,2 s | 2588,9 / 2618 MiB | STL/3MF, watertight, 20.066 triângulos |

Exemplos e relatórios estão em `outputs/validacao-hunyuan/` e `outputs/validacao-triposr/` (artefatos locais, ignorados pelo Git). A geometria Hunyuan na resolução padrão 192 também foi gerada; seu pós-processamento com suavização, decimação e corte foi validado separadamente em `outputs/validacao-hunyuan-192/`. Nesse caso, a recuperação por voxels foi necessária e aumentou a contagem final de triângulos. Esses tempos e picos são medições de uma entrada, não promessa para outras imagens.

Rembg foi testado offline com JPEG sem alpha. Gradio construiu os componentes e respondeu HTTP 200 no loopback; a renderização visual no navegador não foi inspecionada automaticamente. O servidor usado no teste foi encerrado. Os testes pytest incluem malhas sintéticas, o contrato de eixos TripoSR, exportação 3MF e bloqueio de DNS/conexões externas.

Validação final: **13 testes passaram**, `pip check` não encontrou dependências quebradas e `setup.ps1 -WithTripoSR -SkipWeights` concluiu com sucesso, reutilizando os pesos baixados anteriormente. Os três pares STL/3MF de exemplo foram reabertos e verificados como sólidos de 80 mm com base em Z=0.

## Instalação (fase 1)

Instale Python 3.11 x64 e Git. Abra PowerShell nesta pasta:

```powershell
.\setup.ps1
# Ou, com caminho explícito para Python 3.11/3.12:
.\setup.ps1 -Python 'C:\Python311\python.exe'
# Segundo backend:
.\setup.ps1 -WithTripoSR
```

O instalador cria `.venv`, instala PyTorch **2.7.1 cu128** / torchvision 0.22.1, dependências, código oficial e pesos de geometria. Não instala rasterizadores de textura. Faz teste real de multiplicação FP16 na GPU e executa pytest. Downloads podem ocupar vários GB e demorar; há cópias temporárias e de cache adicionais. Mantenha espaço livre em disco e RAM suficiente para offload (32 GB recomendados).

```powershell
.\.venv\Scripts\Activate.ps1
python -m img23d --check-gpu
```

Esperado: GPU `NVIDIA GeForce RTX 5060`, CUDA 12.8 e capability `[12, 0]`. O número CUDA mostrado pelo `nvidia-smi` é a capacidade do driver, não a versão do PyTorch. Falha aqui deve ser resolvida antes de inferência. Não é necessário instalar CUDA Toolkit para o Hunyuan.

O script usa o launcher `py -3.11` por padrão. Se não existir, passe `-Python`. Uma venv existente precisa ter pip (`python -m ensurepip --upgrade`, se necessário).

## Hunyuan3D (fase 2)

```powershell
python -m img23d input.png --height 80 --backend hunyuan --out saida/
```

Esperado: `preprocessada.png`, `modelo.stl`, `modelo.3mf`, `relatorio.json`. A primeira geração pode demorar minutos. O backend usa a classe oficial `hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline`, `from_single_file`, pesos `hunyuan3d-dit-v2-mini/model.fp16.safetensors` e `config.yaml`, 30 passos e resolução 192. Usa FP16, offload de modelos via Accelerate e chunks de 4000 pontos. Carrega inicialmente em CPU para evitar pico ao colocar todos os modelos na GPU.

O limite de 8 GB orienta os padrões, **não constitui uma medição garantida para todas as entradas**. O alocador PyTorch tem teto de 7 GiB (ou 90% da placa, o menor). Feche outros aplicativos de GPU. Se faltar memória, tente `--resolution 128`; se necessário 64. Os logs mostram pico **alocado e reservado pelo PyTorch**, não a VRAM total de outros processos/WDDM. O cache da GPU é liberado entre etapas. A geração não treina modelos.

Compatibilidade de offload: a revisão oficial fixada no setup consulta `pipeline.components`, mas não cria o registro no construtor. O adaptador registra `conditioner`, `model` e `vae` antes de chamar o método oficial, e define o dispositivo dos latentes como CUDA sem mover todos os pesos para GPU. O extrator de superfície usa a API `SurfaceExtractors["mc"]()`.

## Reparo e imprimibilidade (fase 3)

```powershell
python -m pytest -q
python -m img23d input.png --height 80 --smooth 2 --target-faces 50000 --cut-base 0.03
```

Os testes usam esfera com buraco pequeno/grande e componentes desconectados; verificam volume positivo, fechamento, corte, escala, base Z=0, espessura e conteúdo 3MF. O pipeline mantém o componente com maior área, une vértices, remove faces degeneradas/duplicadas, corrige normais e tenta fechar buracos pequenos. Se não houver sólido válido, reconstrói por voxels em CPU e valida novamente. Se ainda falhar, interrompe a exportação.

Escolhas de reparo:

| Opção | Benefício | Limitação |
|---|---|---|
| trimesh + voxels (reparo padrão recomendado) | CPU, controle de resolução e testes simples | Pode alterar detalhes, cavidades e espessuras |
| manifold3d (usado para cortar a base) | Booleanas robustas em sólidos válidos | Não repara arbitrariamente qualquer superfície aberta |
| pymeshlab | Amplo conjunto de filtros | Mais dependências e parâmetros de reconstrução |

`pymeshlab` é instalado porque o pacote oficial Hunyuan o importa; o nosso reparo usa trimesh/scipy/scikit-image. `--voxel-resolution 128` controla o reparo, independentemente de `--resolution` (IA), entre 32 e 256. Lacunas maiores podem resultar em uma casca fechada em vez de reconstruir o interior pretendido; inspecione visualmente.

O corte usa interseção booleana com `manifold3d`, remove uma fração da altura inferior e tampa o plano. Ex.: `0.03` corta 3% **antes** de aplicar a dimensão final. Essa escolha evita falhas de triangulação da tampa observadas em uma malha real com corte por plano. O sólido é validado e, se necessário, reconstruído novamente após suavização/decimação/corte; essa recuperação pode aumentar o número de faces além do alvo de decimação. A aplicação assume Y para cima na saída dos modelos e converte para Z para impressão. Confira a orientação no visualizador.

O relatório inclui watertight, sólido com volume positivo, volume em cm³, dimensões XYZ em mm, triângulos, componentes removidos e tempos/picos de VRAM por etapa. Espessura usa até 512 raios entre superfícies: é **triagem aproximada**, pode perder paredes finas e não substitui análise no fatiador. Partes ligadas por pontes finas, balanços, estabilidade e suportes não são garantidos. Um sólido watertight ainda pode exigir suportes.

## CLI (fase 4)

```powershell
python -m img23d input.png --height 80 --backend hunyuan --out saida/
python -m img23d input.png --width 60 --resolution 128 --out saida-largura/
python -m img23d --help
```

Altura = Z; largura = X. São mutuamente exclusivas; padrão: 80 mm de altura. A base é posicionada em Z=0 e o centro XY na origem. STL não armazena unidades; importe no fatiador como **mm**. 3MF inclui `unit="millimeter"` e geometria Core, sem parâmetros específicos de impressora. Uma execução na mesma pasta substitui os arquivos de saída anteriores.

`--keep-background` desativa rembg (útil para comparar com imagem já preparada); entradas com alpha existente mantêm sua máscara. `--seed` controla a amostragem Hunyuan; os resultados não são necessariamente idênticos entre versões/hardware.

## Gradio (fase 5)

```powershell
python -m img23d --ui
```

Abra `http://127.0.0.1:7860` (ou a porta informada). Envie uma imagem, escolha modelo e altura e clique **Gerar**. Esperado: visualização STL, downloads STL/3MF e relatório. Interface com uma geração por vez; não execute várias CLIs simultaneamente na mesma GPU. Resultados ficam em `outputs/<identificador>/`; remova as pastas manualmente quando não precisar mais delas. O servidor escuta apenas no loopback, sem link público.

## Melhoria — fase 1: seleção do objeto com SAM 2

O modo **Rápido (rembg)** continua sendo o padrão. Opcionalmente, instale o código oficial SAM-2 e o checkpoint **SAM 2.1 Hiera Tiny**:

```powershell
.\setup.ps1 -WithSAM2
# Se o pacote já estiver instalado e faltar somente o peso:
.\.venv\Scripts\python.exe -m img23d.download --sam2-only
.\.venv\Scripts\python.exe -m img23d --ui
```

Reinicie o servidor Gradio após atualizar o código; uma aba já aberta precisa ser recarregada.

1. Envie a imagem e escolha **Escolher objeto (SAM 2)**.
2. Clique **Detectar máscaras (SAM 2)**. Aguarde a imagem com regiões coloridas e números.
3. Clique na região desejada ou escolha o **Número da máscara**. Confira a prévia **Objeto selecionado** antes de gerar. Em regiões sobrepostas, o clique escolhe a menor máscara sob o cursor; o número permite escolher a região maior.
4. Clique **Gerar**. O recorte selecionado mantém seu alpha e não é submetido ao rembg novamente.

Esperado: somente o objeto escolhido entra no pipeline 3D. `objeto_selecionado.png` registra o recorte; `preprocessada.png` mostra a entrada quadrada final. O `relatorio.json` inclui modo de segmentação, número escolhido, tempo e pico de VRAM do SAM. Alterar a imagem ou o modo limpa a seleção. Seleções expiram após 20 minutos de inatividade; nesse caso, detecte novamente.

SAM produz **propostas de regiões**, não nomes/classes de objetos: pode separar partes do mesmo objeto, incluir fundo ou omitir uma região. Por isso a prévia precisa ser conferida. São exibidas até 32 máscaras, com pelo menos 0,5% da área da imagem (mínimo 32 pixels). A imagem é reduzida para no máximo 1024 pixels no maior eixo antes da segmentação; a seleção mantém essa resolução. Não há união manual de máscaras nesta fase.

API confirmada no [README oficial SAM 2](https://github.com/facebookresearch/sam2), [construtor oficial](https://github.com/facebookresearch/sam2/blob/main/sam2/build_sam.py) e [gerador automático](https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py): `build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", checkpoint, ...)` e `SAM2AutomaticMaskGenerator.generate(rgb_uint8)`. O checkpoint local é `models/sam2/sam2.1_hiera_tiny.pt`, do repositório `facebook/sam2.1-hiera-tiny`. O setup fixa o código no commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`; não atualiza automaticamente código já presente.

A arquitetura `Backend` e as bibliotecas existentes foram mantidas. Dependências adicionadas somente para essa opção: pacote oficial **SAM-2**, **hydra-core** e **iopath**. O README upstream recomenda WSL no Windows; esta integração usa Windows nativo com `SAM2_BUILD_CUDA=0`, opção oficial que dispensa a extensão CUDA adicional. O pós-processamento dessa extensão fica desativado. Não há compilação de modelos. SAM usa autocast BF16, recomendado pelo exemplo oficial e suportado na RTX 50; a precisão dos backends 3D não foi alterada.

Configuração conservadora de memória: 32 pontos por lado, 8 por lote e nenhuma camada adicional de recortes; limiares de IoU previsto 0,7 e estabilidade 0,9. O modelo e os embeddings são removidos da GPU no bloco `finally`, inclusive em falhas, e o cache CUDA é limpo antes de liberar a etapa. Apenas imagens e máscaras em CPU permanecem na sessão. SAM e geração compartilham a fila/lock da GPU. O teto PyTorch de 7 GiB continua em vigor; outros aplicativos ainda podem provocar falta de memória. Downloads acontecem somente no setup ou no comando de download explícito.

As melhorias TRELLIS.2 e benchmark/temperatura pertencem às fases 2 e 3 e ainda não foram implementadas.

Validação da fase 1 nesta RTX 5060: **19 testes passaram** e `pip check` passou. SAM 2 offline detectou 32 propostas na foto oficial `notebooks/images/groceries.jpg`, em cerca de 5,2 s de etapa, com 634,6 MiB alocados / 1354 MiB reservados. Depois da liberação, permaneceram somente 8,1 MiB alocados pela infraestrutura CUDA. O fluxo Gradio por API detectou as máscaras, selecionou a máscara 12 e gerou STL/3MF watertight com TripoSR; o relatório está em `outputs/validacao-sam2/ui-relatorio.json`. A renderização interativa do navegador não foi automatizada; o overlay foi inspecionado como imagem, e o callback de clique foi testado. Os testes cobrem também troca de imagem, regiões sobrepostas, alpha, rejeição de seleção inválida e descarregamento do modelo em falhas. O servidor separado utilizado no teste foi encerrado.

## TripoSR (fase 6)

```powershell
.\setup.ps1 -WithTripoSR
python -m img23d input.png --height 80 --backend triposr --resolution 128 --out comparacao/
```

Usa `tsr.system.TSR.from_pretrained`, `stabilityai/TripoSR` (`config.yaml`, `model.ckpt`) e o config indireto `facebook/dino-vitb16`. A rede roda na GPU com autocast FP16; a extração de superfície roda em CPU FP32 para reduzir VRAM e evitar kernels de marching cubes sem suporte a sm_120. Isso é mais lento que o benchmark oficial na A100. O setup altera explicitamente **um import** em `tsr/models/isosurface.py`: troca `torchmcubes.marching_cubes` pelo adaptador `img23d.backends.marching_cubes`, implementado com scikit-image e testado para manter a convenção de eixos do helper oficial. Não altera a rede nem os pesos; dispensa compilação C++/CUDA. A adaptação recusa versões upstream sem o import esperado. `--steps` e `--seed` não afetam TripoSR.

## Cache e operação offline

```powershell
.\setup.ps1 -Cache 'D:\pesos-img23d'
python -m img23d input.png --cache 'D:\pesos-img23d' --height 80
# Repetir somente o download inicial, inclusive pesos opcionais:
python -m img23d.download --cache 'D:\pesos-img23d' --triposr
```

Também aceita `IMG23D_CACHE`. O cache padrão é `models/` relativo ao diretório atual: execute os comandos na raiz do projeto ou informe caminho absoluto. Somente setup e `img23d.download` precisam de internet. Durante geração, flags offline do Hub/Transformers e bloqueio de conexões Python externas impedem downloads/telemetria, permitindo loopback para o Gradio. O bloqueio permanece pelo resto do processo; rode download em outro processo. Para isolamento de todo o processo, inclusive código nativo de terceiros, pode desconectar a rede após setup.

Não há API remota de inferência nem envio de imagens. Rembg usa `u2netp.onnx` em CPU e só inicia se o arquivo estiver presente. TripoSR inclui código local e cache do config DINO. `pesos.json` registra tamanho e SHA256 dos pesos; `installed-requirements.txt` registra versões instaladas. Código upstream é clonado sem atualização automática; guarde também seus commits para reproduzir o ambiente (`git -C models/code/Hunyuan3D-2 rev-parse HEAD`).

## Estrutura e extensão

```
src/img23d/
  backends/       # Backend.generate(PIL.Image) -> trimesh.Trimesh
  preprocess/    # rembg, máscara, recorte, quadrado, margem
  postprocess/   # reparo, escala e relatório
  export/        # STL e 3MF Core
  ui/            # Gradio local
  stages/        # geometria, pós-processamento/escala e textura independentes
  pipeline.py    # orquestração e métricas
  cli.py         # linha de comando
  download.py    # aquisição explícita de pesos
tests/
setup.ps1
```

Para integrar outro modelo, implemente `Backend.generate` e registre a classe em `create_backend`. Devolva coordenadas com Z para cima, sem escala física; reparo, interfaces e exportação são compartilhados.

## Evolução para lotes — fase 1: estágios e descarregamento

Esta fase mantém a CLI, o Gradio, o SAM 2/rembg e a interface `Backend.generate(image)`. Não troca bibliotecas. O pipeline agora coordena estágios independentes em `src/img23d/stages/`:

1. `preprocessamento`: máscara/recorte e imagem quadrada em CPU.
2. `geometria`: cria o backend, gera a malha e fecha/descarta o backend antes de retornar.
3. `posprocessamento`: reparo e escala em mm, em CPU.
4. `textura`: por padrão **ignorada**, sem carregar modelo. Os modos de projeção e Hunyuan Paint serão implementados na fase 4.
5. `relatorio` e `exportacao`: avaliação e saída STL/3MF atuais.

O reparo/escala precede a textura para que mudanças de topologia não invalidem UVs/cores futuros. Somente imagens e malhas com dados CPU cruzam as fronteiras; modelos e embeddings não fazem parte dos resultados. Os backends atuais continuam descarregando seus modelos em `finally`. O estágio de geometria também chama o novo hook opcional `Backend.close()` e descarta a instância, inclusive em falhas. Backends que já implementam apenas `generate()` permanecem compatíveis; um backend futuro que retenha modelos deverá implementar `close()` para liberar esses recursos.

`TextureProcessor.apply(mesh, image)` e `close()` são um ponto de extensão separado para textura; não substituem `Backend`. Nenhum processador de textura é instanciado no fluxo atual. Essa preparação não adiciona GLB/OBJ, cores em 3MF ou textura visível nesta fase.

Após fechar/descartar os modelos, a limpeza executa `gc.collect()` e `torch.cuda.empty_cache()`. Cada entrada de `etapas` no relatório registra:

| Campo | Significado |
|---|---|
| `etapa` | Nome do estágio |
| `estado` | `concluida`, `ignorada` ou `falhou` |
| `segundos` | Duração, incluindo o fechamento do backend no estágio de geometria |
| `pico_vram_mib` / `pico_reservado_mib` | Picos alocado/reservado pelo PyTorch |
| `vram_apos_mib` / `reservado_apos_mib` | Memória após coleta e limpeza do cache |

As falhas também são registradas nos logs e na lista de métricas fornecida à API; `relatorio.json` continua sendo escrito ao concluir a execução. Os nomes antigos `geracao`, `reparo` e `escala` passam a `geometria` e `posprocessamento` (reparo + escala). Relatórios históricos não são alterados. A indicação `textura: "nenhuma"` deixa explícito o comportamento atual. Contexto CUDA e buffers internos podem manter alguns MiB alocados após a limpeza; `empty_cache()` não elimina o contexto nem libera tensores ainda referenciados por outros códigos.

Como testar esta fase:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m img23d input.png --height 80 --backend triposr --resolution 128 --out saida-estagios/
```

Esperado: STL/3MF sem textura, sólido de 80 mm, métricas dos seis estágios e `textura` com estado `ignorada`. Para Hunyuan, troque `--backend triposr` por `--backend hunyuan`. Os testes verificam fechamento e descarte antes da carga do estágio seguinte, limpeza em falhas, pós-processamento independente e preservação do pipeline completo. Reinicie o servidor Gradio para carregar esta atualização.

Os estágios podem ser chamados separadamente em Python, com a mesma lista opcional `records`:

```python
from img23d.stages import generate_geometry, prepare_print_mesh, apply_texture

# image: PIL.Image já preparada; cache: pasta local de pesos.
records = []
mesh = generate_geometry(image, cache, "hunyuan", resolution=128, records=records)
mesh, notes = prepare_print_mesh(mesh, size=80, records=records)
mesh = apply_texture(mesh, image, records=records)  # ignorado nesta fase
```

Modo lote com carga única/retomada, varredura de resoluções, modos concretos de textura, verificação prévia de RAM/VRAM e monitoramento térmico pertencem às fases 2–5. Ainda não foram implementados; o teto de memória e o tratamento de OOM existentes continuam ativos.

Validação real desta fase na RTX 5060, resolução 128 e exemplo `horse.png`: Hunyuan e TripoSR geraram STL/3MF watertight a 80 mm. Picos alocados de geometria: 2250,9 MiB (Hunyuan) e 2588,9 MiB (TripoSR); após o fechamento, ambos ficaram em 8,1 MiB alocados / 20 MiB reservados, mantidos nos estágios seguintes. Relatórios completos: `outputs/validacao-estagios-hunyuan/relatorio.json` e `outputs/validacao-estagios-triposr/relatorio.json`. São medições de uma entrada; não garantem os mesmos picos para outras resoluções/imagens. **27 testes passaram**, incluindo fechamento em falhas/Ctrl+C e registro de pico/limpeza após OOM simulado; `pip check` também passou.

Fontes oficiais consultadas antes da implementação: [README Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), [model card mini](https://huggingface.co/tencent/Hunyuan3D-2mini), [pipeline oficial](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/hy3dgen/shapegen/pipelines.py), [TripoSR](https://github.com/VAST-AI-Research/TripoSR), [API TSR](https://github.com/VAST-AI-Research/TripoSR/blob/main/tsr/system.py), [wheels PyTorch](https://pytorch.org/get-started/previous-versions/). Os pesos e códigos de terceiros mantêm suas próprias licenças; consulte-as para seu uso, inclusive comercial.
