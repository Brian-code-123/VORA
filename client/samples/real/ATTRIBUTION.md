# Real voice samples

Short clips from **MInDS-14** (PolyAI), licensed **CC-BY-4.0**: Gerz et al., *Multilingual and Cross-Lingual Intent Detection from Spoken Data*, 2021. https://huggingface.co/datasets/PolyAI/minds14

Changes: a few clips chosen, written as 8 kHz mono PCM16 (the originals are 8 kHz telephone audio), peak-normalised.

They ask banking questions: run the server with the demo bank KB (`VORA_KB_DIR=eval_kb/bank VORA_INDEX_DIR=index_bank`) to hear answers; with the VORA Box KB they show the refusal path.

| File | Source id | Transcript |
|---|---|---|
| en_us_balance.wav | minds14_en-US_en-US-BALANCE_602baa3abb1e6d0fbce9215b | show me my account balance please |
| en_gb_freeze.wav | minds14_en-GB_en-GB-FREEZE_60268fc8cfa37e1cf217b115 | please freeze my card |
| en_au_atm.wav | minds14_en-AU_en-AU-ATM_LIMIT_response_26 | hello how's this wondering how much money can I withdraw |
| zh_balance.wav | minds14_zh-CN_zh-CN-BALANCE_603542ea4c449c80694dc596 | 请显示我的账户余额 |
| zh_abroad.wav | minds14_zh-CN_zh-CN-ABROAD_603540f1dbcea0ab8733aa04 | 我在国外我需要咯 |
| zh_bill.wav | minds14_zh-CN_zh-CN-PAY_BILL_60363d51dbcea0ab8733ab43 | 我想寄信用卡的 |
