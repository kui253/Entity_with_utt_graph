from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained(
    "/data2/trace/common_param/bart-base"
)
tokenizer = AutoTokenizer.from_pretrained("/data2/trace/common_param/bart-base")
input_str = "{PERSON: A | B | Tom} {DATE: tomorrow | last Monday} {TIME: afternoon}  # A: Hi Tom, are you busy tomorrow’s afternoon? # B: I’m pretty sure I am. What’s up? # A: Can you go with me to the animal shelter?. # B: What do you want to do? # A: I want to get a puppy for my son. # B: That will make him so happy. # A: Yeah, we’ve discussed it many times. I think he’s ready now. # B: That’s good. Raising a dog is a tough issue. Like having a baby ;-)  # A: I'll get him one of those little dogs. # B: One that won't grow up too big;-) # A: And eat too much;-)) # B: Do you know which one he would like? # A: Oh, yes, I took him there last Monday. He showed me one that he really liked. # B: I bet you had to drag him away. # A: He wanted to take it home right away ;-). # B: I wonder what he'll name it. # A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))"
input_str2 = "{PERSON: Emma | Rob | Lauren} {CARDINAL: one | one | 3 | one} {DATE: these days | Christmas | Christmas} {ORG: WOW}  # Emma: I’ve just fallen in love with this advent calendar! Awesome! I wanna one for my kids! # Rob: I used to get one every year as a child! Loved them!  # Emma: Yeah, i remember! they were filled with chocolates! # Lauren: they are different these days! much more sophisticated! Haha! # Rob: yeah, they can be fabric/ wooden, shop bought/ homemade, filled with various stuff # Emma: what do you fit inside? # Lauren: small toys, Christmas decorations, creative stuff, hair bands & clips, stickers, pencils & rubbers, small puzzles, sweets # Emma: WOW! That’s brill! X # Lauren: i add one more very special thing as well- little notes asking my children to do something nice for someone else # Rob: i like that! My sister adds notes asking her kids questions about christmas such as What did the 3 wise men bring? etc # Lauren: i reckon it prepares them for Christmas  # Emma: and makes it more about traditions and being kind to other people # Lauren: my children get very excited every time they get one! # Emma: i can see why! :)"
input_str3 = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([input_str3], max_length=1024, return_tensors="pt")
summary_ids = model.generate(
    inputs["input_ids"], num_beams=2, min_length=0, max_length=20
)
output = tokenizer.batch_decode(
    summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
pass
