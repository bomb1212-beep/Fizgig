"""The shared prompt builder: the character's age and adult proportions, traits only where they can show.

  venv\\Scripts\\python.exe -B -m unittest pipeline.test_pipeline_prompting      (from automation\\)
"""
import unittest

from pipeline import batch_gen, epochs, experiment, prompting, seed

CARD = {"trigger": "suamom", "caption_base": "korean webtoon style", "subject": "the woman",
        "fixed_traits": ["low ponytail"], "face_traits": ["small mole under her eye"],
        "outfits": {"default": "navy top"}}
FULL = {"half": "body", "shot": "full body", "view": "front view"}
UPPER_BACK = {"half": "body", "shot": "upper body", "view": "back view"}
COWBOY = {"half": "body", "shot": "cowboy shot", "view": "side profile"}
CLOSE = {"half": "face", "shot": "close-up", "view": "front view"}


def with_age(age, **profile):
    return {**CARD, "profile": {"age": age, **profile}}


class BuildTests(unittest.TestCase):
    def test_without_an_age_the_prompt_is_the_old_one(self):
        card = {k: v for k, v in CARD.items() if k != "face_traits"}
        self.assertEqual(prompting.build(card, "smile"),
                         "suamom, korean webtoon style, low ponytail, navy top, smile")

    def test_an_adult_age_names_the_age_and_the_proportions_on_body_shots(self):
        card = with_age(28)
        full = prompting.build(card, "full body", FULL)
        self.assertTrue(full.startswith("suamom, korean webtoon style, adult woman, 28 years old, mature face, "
                                        + prompting.ADULT_PROPORTIONS), full)
        close = prompting.build(card, "close-up", CLOSE)
        self.assertIn("adult woman, 28 years old, mature face", close)
        self.assertNotIn(prompting.ADULT_PROPORTIONS, close)

    def test_the_gender_comes_from_the_profile_before_the_subject(self):
        self.assertIn("adult man, 40 years old", prompting.build(with_age(40, gender="man"), "x"))
        self.assertIn("adult woman, 28 years old", prompting.build(with_age(28), "x"))
        no_subject = {k: v for k, v in with_age(30).items() if k != "subject"}
        self.assertIn("adult, 30 years old", prompting.build(no_subject, "x"))

    def test_a_minor_is_never_called_adult(self):
        prompt = prompting.build(with_age(12), "x", FULL)
        self.assertIn("woman, 12 years old", prompt)
        self.assertNotIn("adult", prompt)

    def test_a_bad_age_is_no_age(self):
        for bad in ("", None, "old", 0, -3, True):
            self.assertEqual(prompting.profile(with_age(bad))["age"], None, bad)
        self.assertEqual(prompting.profile(with_age("28"))["age"], 28)

    def test_face_traits_only_where_the_face_is_big_enough(self):
        card = with_age(28)
        self.assertIn("small mole under her eye", prompting.build(card, "x", CLOSE))
        self.assertIn("small mole under her eye", prompting.build(card, "x", COWBOY))
        self.assertIn("small mole under her eye", prompting.build(card, "x"))          # an eval prompt
        self.assertNotIn("small mole under her eye", prompting.build(card, "x", FULL))
        self.assertNotIn("small mole under her eye", prompting.build(card, "x", UPPER_BACK))
        for cell in (FULL, UPPER_BACK, CLOSE):
            self.assertIn("low ponytail", prompting.build(card, "x", cell))     # fixed traits always

    def test_the_build_words_go_last(self):
        card = with_age(28, build="tall, slender")
        prompt = prompting.build(card, "from side", COWBOY)
        self.assertTrue(prompt.endswith("navy top, from side"), prompt)
        self.assertIn("tall, slender", prompt)

    def test_a_card_without_an_age_is_warned_about(self):
        self.assertTrue(prompting.warnings(CARD))
        self.assertEqual(prompting.warnings(with_age(28)), [])


class EveryGeneratorUsesItTests(unittest.TestCase):
    def test_batch_b_the_epoch_grid_and_m0_ask_for_the_same_character(self):
        card = with_age(28)
        cell = {**CLOSE, "tags": "close-up, face focus"}
        expected = prompting.build(card, "close-up, face focus", cell)
        self.assertEqual(batch_gen.prompt_for(card, "cmp_closeup_front_neutral", cell), expected)
        self.assertEqual(epochs.prompt_for(card, {"prompt": "close-up, face focus"}),
                         prompting.build(card, "close-up, face focus"))
        self.assertEqual(experiment.prompt_for(card, {"prompt": "smile"}), prompting.build(card, "smile"))


class CaptionPolicyTests(unittest.TestCase):
    def test_omit_strips_face_traits_from_captions_too(self):
        card = {**with_age(28), "caption_traits": "omit"}
        self.assertIn("small mole under her eye", seed.trait_terms(card))
        caption = seed.add_traits("suamom, smile, small mole under her eye, low ponytail", card)
        self.assertNotIn("mole", caption)
        self.assertNotIn("ponytail", caption)
        self.assertNotIn("28 years old", seed.card_caption(card))

    def test_include_appends_them(self):
        card = {**with_age(28), "caption_traits": "include"}
        self.assertIn("small mole under her eye", seed.card_caption(card))
        self.assertIn("small mole under her eye", seed.add_traits("suamom, smile", card))


class FieldTests(unittest.TestCase):
    def test_typed_fields_round_trip_through_the_card(self):
        card = prompting.apply(CARD, {"age": " 28 ", "fixed_traits": "low ponytail, black scrunchie ,",
                                      "face_traits": "small mole under her eye"})
        self.assertEqual(card["profile"], {"age": 28})
        self.assertEqual(card["fixed_traits"], ["low ponytail", "black scrunchie"])
        self.assertEqual(prompting.fields(card), {"age": "28", "fixed_traits": "low ponytail, black scrunchie",
                                                  "face_traits": "small mole under her eye"})
        self.assertNotIn("profile", CARD)                       # a copy: the card given is not touched

    def test_an_empty_age_removes_it_and_keeps_the_rest_of_the_profile(self):
        card = prompting.apply(with_age(28, gender="woman"), {"age": ""})
        self.assertEqual(card["profile"], {"gender": "woman"})

    def test_a_bad_age_is_refused(self):
        for bad in ("twenty", "0", "200", "2.5"):
            with self.assertRaises(prompting.ProfileError, msg=bad):
                prompting.apply(CARD, {"age": bad})


if __name__ == "__main__":
    unittest.main()
