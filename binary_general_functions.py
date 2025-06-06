# Import required packages
import pandas as pd
import os
import re
import csv


def get_instruction():
    '''
    Returns the instruction part of the prompt
    '''
    instruction = (
        '''###Task:
        You are a narrative analyst and you will label passages using 3 different structural elements.
        You are going to analyze and identify whether the text contains the narrativity structural elements 'agency', 'event sequencing' and 'world making'.
        Then you will also decide whether the text contains a story or not.

        ### Instructions
        Think step by step.

        First, consider 'Agency' using the following statement:
        “This passage foregrounds the lived experience of particular agents.”
        If you find any evidence supporting this statement, mark it as 1.
        If there is no such evidence, mark it as 0.

        Next, consider 'Event sequencing' using the following statement:
        “This passage is organized around sequences of events that occur over time.”
        If you find any evidence supporting this statement, mark it as 1.
        If there is no such evidence, mark it as 0.

        Then, consider 'World making' using the following statement:
        “This passage describes a location or a place at which something occurs.”
        If you find any evidence supporting this statement, mark it as 1.
        If there is no such evidence, mark it as 0.

        Finally, decide whether the passage contains a story.
        If the passage contains a story, even a simple one, mark it as 1.
        If the passage does not contain a story, mark it as 0.

        If the answer is uncertain for any element, prefer 1 if there is some evidence.

        ### Output
        Respond strictly in the following structure containing either a 0 (no) or 1 (yes) for agency, event sequencing, world making, and story, without further explanation:
        “Agency: 0 or 1
        Event sequencing: 0 or 1
        World making: 0 or 1
        Story: 0 or 1”

        ### Passage to evaluate\n
        '''
        )
    return instruction


def extract_degrees(text):
    '''
    Returns the numerical degrees for each element from the model's output
    '''
    agency = re.search(r"agency.*?(\d)", text, re.IGNORECASE)
    event_seq = re.search(r"event sequencing.*?(\d)", text, re.IGNORECASE)
    world_making = re.search(r"world making.*?(\d)", text, re.IGNORECASE)
    story = re.search(r"story.*?(\d)", text, re.IGNORECASE)

    return [
        int(agency.group(1)) if agency else None,
        int(event_seq.group(1)) if event_seq else None,
        int(world_making.group(1)) if world_making else None,
        int(story.group(1)) if story else None
    ]


def get_data():
    '''
    Returns the input data
    '''
    print('Loading input data...')
    df = pd.read_csv("threads1000_format_preprocessed.csv")
    print(f"{len(df)} passages to process.")

    return df


def create_outfile(output_csv):
    '''
    Creates the output file
    '''
    if not os.path.exists(output_csv):
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'name', 'agency', 'event_sequencing', 'world_making', 'story'])


def write_results(degrees, name, output_csv, output):
    '''
    Writes the results to the output file
    '''
    if None not in degrees:
        results_row = [name] + degrees
        pd.DataFrame([results_row], columns=[
            'name', 'agency', 'event_sequencing', 'world_making', 'story'
        ]).to_csv(output_csv, mode='a', index=False, header=False)
    else:
        print(f"Parsing failed for {name}:\n{output[0]}\nExtracted: {degrees}")


def get_example(n):
    '''
    Return a given amount of examples
    (max 3, as there are 3 examples in the list)
    '''
    all_examples = [
        (
            '''# PASSAGE:
            "The title says most of what I want to say, but to elaborate a bit, there seems to be a lot of outrage over the fact that Darren Wilson was not indicted, not just among members of the community but abroad, and I'm not quite sure where it's coming from.  Initially, it seemed as though Brown may well have been fleeing from the scene and was shot without posing a threat, but as new information has been released that seems less and less to have been the case.  Part of me thinks the outrage is largely due to the fact that people have associated Mike Brown with a greater (and admittedly perfectly legitimate) problem, namely that of police targeting of black youth.  Now that he's so strongly associated with that problem, it seems like people don't want to admit that perhaps Brown did at least reasonably seem to pose a threat because it would feel like they're betraying a cause they care a lot about, even if to admit that would be the more reasonable position to take.  Change my view."

            # OUTPUT:
            “Agency: 1
            Event sequencing: 1
            World making: 0
            Story: 1”'''
        ),
        (
            '''# PASSAGE:
            "I was about to post exactly the same thing as you.  Over is the superior method when you live by yourself or only with adults. As soon as you introduce children or animals, under is better. Not only does it prevent the unrolling of a roll on the floor but it also slows down your child from using the entire roll when they're being potty trained.  Under provides substantially more control."

            # OUTPUT:
            “Agency: 0
            Event sequencing: 0
            World making: 0
            Story: 0”'''
        ),
        (
            '''# PASSAGE:
            "SPOILERS ALL:  Through the many seasons of Supernatural, Sam has shown himself to be a self centered, whiny, crybaby. Dean Winchester sacrificed his personal happiness a million times to save Sam's whiny ass. He sacrificed his own education and held down the family business while Sam ran off to party at college. Dean made deals with demons to save Sam after he was dumb enough to get himself killed. In return, Sam watches Dean get eaten by a Hellhound, die, go to Hell, and doesn't lift a finger to save him.   Add. Rinse. Repeat, through several more seasons (far too many examples of Selfish Sam to list) and we have (yet again) Dean ""die"" heroically killing the Boss monster (while Sam is off playing with coffee creamer somewhere) and ending up trapped in Purgatory for over a year. And what does Sam do? He get's himself a dog and a girlfriend and basically says, ""Dean Who?"" Then when Dean gets himself out by teaming up with a friendly vampire, all big meat headed Sam can think about is trying to kill the vamp who helped rescue Dean.  And most heinous of all, the very first time Dean lets Sam drive Baby, Sam crushes her under a semi.  Dean should ditch Sam and his pretty hair on some roadside and go hunting with Castiel from now on."

            # OUTPUT:
            “Agency: 1
            Event sequencing: 1
            World making: 1
            Story: 1”'''
        )
    ]

    examples_text = "\n\n".join(all_examples[:n])
    return f'''\n### Examples
    Follow the following example{'s' if n > 1 else ''}, they contain a passage and its corresponding binary classification of the elements:

    {examples_text}'''
