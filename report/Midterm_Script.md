# Mid-term presentation — FINAL delivery script

**Parv: slides 1–6 — 893 words, 5:00** (unchanged; this is the version you timed)
**Mayank: slides 7–13 — 600 words, 4:40**

Budgets are set from your *measured* speaking rates, not a generic one: Parv runs at
about 180 words a minute, Mayank at about 130. That difference is the whole reason
the halves came out uneven — not effort. So Mayank's section is two-thirds the word
count of Parv's, and it should land in the same five minutes.

The slides are unchanged.

Timings are spoken content only. `[...]` are stage directions.
Lines marked ✂ are safe to drop live if you are behind.

---
---

# PARV — slides 1 to 6 — target 5:00

---

## Slide 1 — Title · 0:40

Good morning. Our project is on motor subtype progression in Parkinson's disease,
using the PPMI dataset.

There's a standard clinical rule that labels a patient as tremor-dominant or PIGD
at a given visit. That rule works on one visit at a time, and it works well for
that. What we wanted to understand is what happens *across* visits — because when
a patient's recorded subtype changes, that could be the disease progressing, or it
could just be the examination coming out differently that day.

Separating those two things is the whole project.

---

## Slide 2 — One number, cut into three · 0:50

The subtype comes from a single number. Eleven tremor items, five gait and postural
items, take the two averages and divide. Above 1.15 you're tremor-dominant, below
0.90 you're PIGD, and in between you're Indeterminate.

[*point at the bar*]

The thing to notice is this picture. That middle band is 0.25 units wide — but the
ratio itself runs from zero to about eleven. So the band is very narrow relative to
the scale it sits on. Please hold onto that, because it comes back twice.

And the rule is a snapshot. It tells you where a patient is today. It has nothing
to say about whether a change between two visits is real.

---

## Slide 3 — The dataset we built · 0:45

We didn't take the subtype column off the shelf — we rebuilt it from the raw item
scores, so that we controlled every decision. Sporadic Parkinson's only, and only
OFF-medication examinations, because that's what the standard definition uses.

That gives about a thousand patients and four and a half thousand labelled visits.
For anything longitudinal you need at least two labelled visits, and that's 934
patients. That number is the cohort for everything that follows.

✂ We validated our labels against PPMI's own — we agree on every single visit for
the two-class column, and on 99.87% for an archived three-class one.

And the box on the right. "Unable to Rate" is recorded as the number 101, sitting
in the same column as a real score of 0 to 4. If you average that in, one entry
pushes the PIGD score to about twenty, and the patient gets labelled PIGD when
they're actually tremor-dominant. There are over three thousand of those on the
postural stability item alone. Finding that early is a large part of why we trust
the rest.

---

## Slide 4 — What the records actually show · 0:50

This is just counting — no model yet. Take every pair of consecutive labelled
visits and ask where the patient went.

Two things jump out. The middle row: if a patient is Indeterminate today, there's
only a 19% chance they're Indeterminate at the next visit. They're much more likely
to be somewhere else. So Indeterminate isn't really behaving like a category — it
looks more like a region patients pass through.

And PIGD isn't a one-way door either. Almost 30% of PIGD visits are followed by
something that isn't PIGD, and 18% go straight back to tremor-dominant. People
often think of PIGD as the endpoint; the record doesn't support that.

But — and this matters — there is still real drift. Changes towards PIGD outnumber
the other direction 253 to 141, and that's highly significant. So something genuine
is going on. It's just buried in a lot of movement.

---

## Slide 5 — A single recorded change is weak evidence · 1:10
### ⚑ Key slide — do not rush this one

This slide is the heart of the motivation, so I'll spend a moment on it.

A third of all the subtype changes we observe reverse immediately — at the very
next visit the patient is back where they were. That is not what progression looks
like.

Second number. If we take apparent conversions to PIGD and add one requirement —
the new subtype has to still be there at the next visit — half of them disappear.
263 becomes 127.

And the reason is the third number. Within a single patient, the ratio bounces
around with a standard deviation of 1.22, and the band we're asking it to sit
inside is 0.25 wide. It's about five times too noisy for the question we're asking
it. In 73% of consecutive visits, a patient's own ratio moves further than the
entire width of the middle band.

So if you happen to sit near a threshold, you will cross it now and then for
reasons that have nothing to do with your disease.

That's the problem we needed a model for.

---

## Slide 6 — The simple model failed · 1:00

We didn't jump straight to a complicated model. We started with the simplest thing:
a Markov chain on the observed label. That model says the current label is all you
need — where the patient came from shouldn't matter.

So we tested exactly that. Take every visit where the patient is currently labelled
TD, and split them by where they came from. If they arrived at TD from TD, there's
an 8.6% chance they're PIGD next visit. If they arrived at TD from PIGD, it's 39.3%.

Same label today. Four and a half times the risk tomorrow.

Chi-squared is 156, and it's not just TD — the test rejects for all three labels.

So the observed label is not a complete description of the patient. Something is
being remembered that the label doesn't capture. And that is precisely the pattern
you get when there's a real underlying state and you're only seeing a noisy reading
of it.

That's what took us to the hidden Markov model — not because it was next in a
textbook, but because the data told us the simpler one was wrong.

[*handover*] I'll hand over to Mayank.

### ⏱ Checkpoint: hand over at **5:00**.

---
---

# MAYANK — slides 7 to 13 — target 4:40

> **Read this before you practise.**
> You speak at about 130 words a minute — that is a good, clear pace and you should
> not try to speed up. Instead this script is written to *your* rate: it is 600
> words, which lands at 4:40. The previous one was 900, which is why it came out
> at seven minutes.
>
> Three rules:
> 1. **Never read a table aloud.** Say what it means; they can read.
> 2. **One sentence, one idea.** Full stop, breathe, next.
> 3. **If you fumble a number, carry on.** Don't restart the sentence — the restarts
>    are where the two minutes went.

---

## Slide 7 — Separating the state from the label · 0:45

Thanks. Here's the idea.

There's an underlying motor state — the top row. It moves gradually, and we never
see it.

What we see is the bottom row. That's what the examination wrote down.

[*point at the third column*]

Third column. The state is Indeterminate. The exam recorded PIGD.

So the model estimates two things at once. How fast the real state changes. And how
reliably it gets recorded. The earlier models couldn't express that second part.

One caveat. We impose a minimum recording reliability, or the fit degenerates.
That's our choice, not the data's.

---

## Slide 8 — What the model estimates · 0:50

Three results — and these are the model's estimates, not measurements.

First, it recovered an ordering we never gave it. Patients move *through*
Indeterminate. Going straight from TD to PIGD is sixteen times rarer. PIGD back to
TD comes out at zero.

Second, reliability. TD is recorded right 94% of the time. PIGD, 84%. Indeterminate,
50% — and to be precise, **50% is the floor we imposed**, so it's an upper bound.
Unconstrained, the data pushes it to 0.36. Essentially uninformative.

Third, the key one. The labels suggest a TD patient stays TD about three and a half
years. The model says the underlying state lasts around ten.

So most of the churn is measurement, not the patient.

---

## Slide 9 — Validation · 0:55
### ⚑ Key slide — the examiners will focus here

This is the strictest validation we can do.

We hold out one patient completely. Refit on the other 933. Then predict their
visits one at a time — and to predict a visit, the model only sees what came before.

That's 3,361 predictions, at **75.4%**.

Now the honest reading. The HMM wins, but by under a point over a plain Markov
chain, and about three over just saying nothing changes. The real separation is the
Brier score, which judges the whole distribution rather than the top choice.

And the caveat that matters most. We predict the next label *our own rule* produces.
That's a noisy record. So this is agreement with a record — not accuracy about
disease progression, and not clinical validation.

---

## Slide 10 — Two stress tests · 0:45

Then we tried to break that number. Two ways.

First, we narrowed the Indeterminate band. Accuracy rose, to 78%. But the
persistence baseline rose more — so our advantage over "nothing changes" halved.
We'd only deleted the hardest class. The number improved; the model didn't.

Second, five latent states instead of three. No better. Only 404 of 934 folds
converged, and six starting points gave six different answers. That model isn't
identified by this data.

Two negative results, one lesson. Accuracy alone is a misleading way to compare
these models. That's why we quote the gap to the baseline.

### ⏱ Checkpoint: leave this slide at about **8:20**.

---

## Slide 11 — The continuous ratio · 0:30

We also built the proposal's Bayesian growth model, on the continuous ratio.

It gives one thing the HMM can't — a progression rate in clinical units. About 0.09
a year.

It loses on prediction, for the same reason as everything else: its spread is about
five times the width of the band.

So we keep both. And the box at the bottom is our overall reading — the noise and
the signal are both real, at comparable size.

---

## Slide 12 — Where this goes next · 0:40

Four directions.

First, the ratio throws away severity. Two over four and ten over twenty are both
0.5 — very different patients. So we want to model tremor and gait separately. That
also recovers 725 visits where the ratio doesn't exist at all.

Second, every model here assumes a constant rate. We want to compare early and late
follow-up. That's a hypothesis, not a result.

Third, whether patients fall into distinct progression patterns at all.

Fourth, understanding *when* the model fails. The five-state result is why that
matters.

---

## Slide 13 — Close · 0:15

To close. Subtype progression is a gradual underlying process, seen through an
imperfect measurement.

So a single recorded change is weak evidence. A sustained one is much stronger.

Thank you — happy to take questions.

---
---

# Timing sheet

| Slide | Who | Target | Cumulative |
|---|---|---|---|
| 1 Title | Parv | 0:40 | 0:40 |
| 2 One number | Parv | 0:50 | 1:30 |
| 3 Dataset | Parv | 0:45 | 2:15 |
| 4 Records | Parv | 0:50 | 3:05 |
| 5 Weak evidence | Parv | 1:10 | 4:15 |
| 6 Markov failed | Parv | 1:00 | **5:15 → handover** |
| 7 State vs label | Mayank | 0:45 | 6:00 |
| 8 What it estimates | Mayank | 0:50 | 6:50 |
| 9 Validation | Mayank | 0:55 | 7:45 |
| 10 Stress tests | Mayank | 0:45 | **8:30** |
| 11 Continuous ratio | Mayank | 0:35 | 9:05 |
| 12 Future work | Mayank | 0:45 | 9:50 |
| 13 Close | Mayank | 0:15 | **10:05** |

**Two checkpoints to watch on a phone timer:**
- Handover at **5:00–5:15**. If Parv is past 5:30, he drops the ✂ line on slide 3.
- Leaving slide 10 at **8:15–8:30**. If Mayank is past 8:45, he drops the ✂ line on
  slide 11 and delivers slide 12 as four short headlines only — one sentence each,
  no elaboration.

---

# If Mayank is still over after practising

Do these in order.

1. **Slide 11 becomes three sentences.** "We also built the proposal's growth model.
   It gives a progression rate in clinical units — about 0.09 a year — which the HMM
   can't. It loses on prediction, so we keep both for different reasons." Saves ~20 s.
2. **Slide 12 becomes headlines.** Say only the bolded phrase from each of the four
   rows, plus one clause. Saves ~20 s.
3. **Move slide 7 to Parv.** It follows directly from his slide 6 — "that's what took
   us to the hidden Markov model, and here's how it works" — so it's a natural join,
   and Parv has proven he holds time. This moves 45 s across and leaves Mayank six
   slides instead of seven.

Option 3 is the biggest single lever. Only use it if 1 and 2 aren't enough, because
it means re-rehearsing the handover.

---

# Three numbers Mayank must not fumble

- **934 patients / 3,361 predictions / 75.4%**
- **+3.1 → +1.2** — the margin over baseline collapsing when the band narrows
- **404 of 934** folds converged with five latent states

# Two language traps

- Slide 8: say **"underlying state"** or **"latent state"** — never "true disease state".
- Slide 9: say **"agreement with the next recorded label"** — never "accuracy at
  predicting progression".

If you're asked something you don't know, say so and say how you'd test it. "We
haven't tested that, but here's how we would" is a perfectly good answer at a
mid-term.
