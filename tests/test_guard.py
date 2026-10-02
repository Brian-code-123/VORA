import pytest

from vora.guard import best_sentence, is_refusal, polarity_conflict

CANTONESE_EN = "Languages: VORA Box understands and speaks Mandarin Chinese and English. Cantonese speech is not supported."
CANTONESE_ZH = "语言：VORA Box 能听懂并说普通话和英语，也能回答中英文混合的句子。不支持粤语。"


def test_conflict_cantonese_en():
    assert polarity_conflict("does it understand cantonese", CANTONESE_EN, "Yes, the VORA Box understands") is True


def test_conflict_cantonese_zh():
    assert polarity_conflict("支持粤语吗", CANTONESE_ZH, "支持粤语") is True


def test_no_conflict_when_answer_is_also_negative():
    assert polarity_conflict("does it understand cantonese", CANTONESE_EN, "No, Cantonese is not supported") is False


def test_no_conflict_when_negation_unrelated():
    chunk = "Wi-Fi is only needed for firmware updates. It does not need a cloud account. The X200 supports Wi-Fi 6."
    assert polarity_conflict("does the x200 support wi-fi 6", chunk, "Yes, the X200 supports Wi-Fi 6") is False


def test_no_conflict_non_yes_no_question():
    assert polarity_conflict("what languages does it support", CANTONESE_EN, "Mandarin and English") is False


def test_double_negation_not_flagged():
    chunk = "It is not true that Cantonese is unsupported: Cantonese is not unsupported on the X200."
    assert polarity_conflict("does it understand cantonese", chunk, "Yes it does") is False


@pytest.mark.parametrize("head,expected", [
    ("The context does not provide information about", True), ("I cannot find that in the context", True),
    ("上下文没有提供相关信息", True), ("无法回答", True),
    ("The warranty is two years", False), ("保修期是两年", False), ("", False),
])
def test_refusal_phrases_en_zh(head, expected):
    assert is_refusal(head) is expected


def test_best_sentence_prefers_overlap():
    chunk = "Operating conditions: use between 0 and 40 degrees. Keep it away from heaters. Humidity must be 10 to 90 percent."
    assert best_sentence(chunk, "what humidity can it take").startswith("Humidity must be")
    assert best_sentence("第一句。第二句关于保修。", "保修期") == "第二句关于保修。"
    assert best_sentence("Only one sentence", "anything") == "Only one sentence"


from vora.guard import extractive_answer, is_digitish, is_yes_no_question, numbers_mismatch


def test_numbers_mismatch_catches_invented_figures():
    assert numbers_mismatch("The X200 costs 199 USD.", "the x200 is priced at $1,000") is True
    assert numbers_mismatch("range of 3 meters", "the m100 can hear you within 30 days") is True
    assert numbers_mismatch("The X200 costs 199 USD.", "the x200 costs 199") is False
    assert numbers_mismatch("two years", "it is two years") is False        # words are never digits
    assert numbers_mismatch("Warranty 2 years", "about 2.0 years") is False  # 2.0 == 2
    assert numbers_mismatch("价格 199 美元", "售价 199 美元") is False
    assert numbers_mismatch("The VORA-X200 warranty is 2 years.", "The VORA-X200 is 2 years") is False   # model code digits ignored
    assert numbers_mismatch("The VORA-X200 costs 199 USD.", "The X200 costs 19 dollars") is True


def test_refusal_phrase_in_the_context():
    assert is_refusal("The support email is provided in the context") is True


def test_yes_no_question_detector():
    for q in ("does it understand cantonese", "is the m100 loud", "can i use it", "支持粤语吗", "有没有蓝牙"):
        assert is_yes_no_question(q), q
    for q in ("how long is the warranty", "what is the wake word", "x200保修期多久"):
        assert not is_yes_no_question(q), q


def test_is_digitish_tokens():
    for t in ("1", ",", "000", " 199", "$", "$1,000", "2.4"):
        assert is_digitish(t), t
    for t in ("years", " the", "USD", "", "X200"):
        assert not is_digitish(t), t


def test_best_sentence_does_not_split_inside_an_email_or_version():
    chunk = "Support contact: email support@vora.example. Support hours are Monday to Friday, 9:00 to 18:00."
    assert best_sentence(chunk, "what is the support email") == "Support contact: email support@vora.example."
    assert best_sentence("Firmware 1.2.3 is current. Other text here.", "what firmware version").startswith("Firmware 1.2.3")


from vora.guard import expects_number, has_number


def test_expects_number_and_has_number():
    for q in ("how far can the m100 hear me", "how long is the warranty", "how much does it cost", "x200 拾音范围", "保修期多久", "多少钱"):
        assert expects_number(q), q
    for q in ("what is the wake word", "does it support wifi", "怎么重置"):
        assert not expects_number(q), q
    for t in ("within 3 meters", "two years", "五米", "199 USD"):
        assert has_number(t), t
    for t in ("a wide range", "拾音范围是麦克风拾音范围内。"):
        assert not has_number(t), t


def test_has_number_ignores_model_code_digits():
    assert not has_number("The M100 and the X200 are loud")
    assert has_number("The M100 reaches 3 meters")


def test_yes_no_without_negation_quotes_the_two_best_sentences_in_order():
    chunk = "Factory reset: hold the reset button for 10 seconds. This erases all settings and imported documents. Firmware is kept."
    out = extractive_answer("will a factory reset delete the firmware", [chunk])
    assert out.startswith("Factory reset") and out.endswith("Firmware is kept.")


def test_yes_no_with_negation_quotes_only_the_negating_sentence():
    out = extractive_answer("does it understand cantonese", [CANTONESE_EN])
    assert out == "Cantonese speech is not supported."


def test_non_yes_no_still_one_sentence():
    chunk = "Support contact: email support@vora.example. Support hours are Monday to Friday."
    assert extractive_answer("what is the support email", [chunk]) == "Support contact: email support@vora.example."
