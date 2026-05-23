# Defense-Lab — Resultados e tese (segmentação operacional de uso-de-solo)

**Backbone:** SAM2 (promptável, zero-shot) · **Hardware:** GH200 / bf16 · **Escopo:** segmentação
de feições operacionais de uso-de-solo em sensoriamento remoto (drone/satélite). Classe-agnóstico;
sem rastreamento/identificação de pessoas.

## Tese
> O gargalo operacional da segmentação RS com foundation models **não é a qualidade da máscara** —
> SAM2 zero-shot já é forte — **é a localização automática de baixo falso-positivo.**

Os experimentos abaixo (4 datasets reais) sustentam isso com número.

## 1. SAM2 zero-shot é um baseline promptável forte (sem treino)
IoU de máscara dado um prompt de caixa, por dataset:

| dataset | tipo | IoU (box, zero-shot) |
|---|---|---|
| VDD | drone, land-cover | 0.67 |
| Satélite (saidines12) | satélite, binário | 0.81 |
| Morocco buildings | satélite, edificações | 0.70 |
| Airstrip (S1-AAD) | linha fina (SAR) | 0.15 — fora de encaixe |

→ Máscara de boa qualidade sem nenhum treino, em 3 domínios distintos.

## 2. Adaptação leve (LoRA / prompt-tuning) NÃO generaliza
Mesmo protocolo box-prompted, IoU antes→depois:

| método | VDD | Satélite | Morocco | robusto? |
|---|---|---|---|---|
| LoRA (decoder) | +3.0 (3-seed, ±0.6) | −0.5 | −1.5 | não (1/4) |
| Prompt-tuning (8 tokens) | −4.9 | +1.5 | +0.3 | não (1/3) |

→ Ganhos específicos de dataset / dentro do ruído. Vanilla-PEFT overfita em RS pouco-dado
(consistente com a literatura, ex. SAMed). **Não sustentamos claim de "adaptação melhora".**

## 3. ACHADO-CHAVE — sem oráculo, localização é o gargalo
AMG propõe máscaras (sem caixa de GT) → casamento com GT → métricas operacionais:

| dataset | recall@0.5 | precision@0.5 | FP / imagem | IoU (matchados) |
|---|---|---|---|---|
| VDD | 0.27 | 0.09 | 65.5 | **0.83** |
| Satélite | 0.53 | 0.18 | 14.1 | **0.74** |
| Morocco | 0.34 | 0.06 | 44.5 | **0.70** |

→ **Quando acha, a máscara é ótima (0.70–0.83). Mas perde a maioria (recall 0.27–0.53) e inunda
de falsos-positivos (14–66/img).** O problema é localização + FP, não máscara.

**Curva de operação (filtro de score no AMG):** subir o threshold de confiança apenas **troca recall
por FP** — nunca há ponto bom. Para FP<10 o recall cai pra ~0; a precision satura em **~0.25–0.30** em
qualquer recall útil; o IoU dos matchados segue alto (0.68–0.87) o tempo todo. → **filtro de score NÃO
resolve a localização**; é preciso um **localizador aprendido** (detector / prompt-generator), não as
propostas automáticas do foundation model.

## Contribuições
1. **Caracterização**: SAM2 como baseline promptável forte zero-shot em RS (4 datasets).
2. **Achado quantitativo**: o gargalo é localização automática low-FP, não qualidade de máscara.
3. **Data-engine CC+VLM**: vídeo Creative-Commons → frames → SAM2 (recorta) → Claude/Haiku (classifica)
   → 951 instâncias rotuladas, automático e barato.
4. **Limite de encaixe**: feição linear fina (pistas em SAR) → promptable-seg é a ferramenta errada;
   usar detecção + métrica de detecção.

## Próximos passos (onde está o valor / a oportunidade soberana)
- **Localizador aprendido** (detector / prompt-generator low-FP estilo RSPrompter), não o decoder
  nem filtro de score (provado insuficiente). É aqui a contribuição soberana.
- Métrica operacional: recall/precision + **taxa de falso-positivo** (relevante p/ vigilância/SSA).
- Domínios com headroom real (SAR fino, alvo faint).

## Reprodutibilidade
Cada experimento salva config/seed/ambiente/métricas/figuras em `experiments/`. Scripts:
`scripts/landuse_experiment.py`, `lora_hf.py`, `prompt_tuning_xdataset.py`, `no_oracle_eval.py`,
`youtube_ingest.py` (data-engine), `build_airstrip_dataset.py`. Dashboard: `scripts/dashboard.py`.
