
# Add this to your training configuration:

characters=CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    characters=" !'(),-.:;?abcdefghijklmnoprstuvyæñāēīōśşūǣḍḥḷṁṅṇṉṛṝṭ",
    punctuations=",):(;-?'.!",
    phonemes=None,
    is_unique=True,
    is_sorted=True,
)
