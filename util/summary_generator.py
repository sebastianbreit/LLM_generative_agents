class SummaryGenerator:
    def __init__(self, summary_pipeline):
        self.pipeline = summary_pipeline

    def generate_summary(self, dialogue, min_length=30, max_length=128, do_sample=False):
        # Call pipeline
        summary = self.pipeline(dialogue, max_length=max_length, min_length=min_length, do_sample=do_sample)
        return summary

    def generate_list_of_sentences(self, summary):
        # Split at each dot
        sentence_list = summary.split('.')

        # Append dot and remove blank
        sentence_list = [sentence[1:] + '.' if sentence.startswith(' ') else sentence + '.' for sentence in
                         sentence_list]

        return sentence_list

    def generate_memories(self, dialogue, min_length=30, max_length=128, do_sample=False):
        # Get summary
        summary = self.generate_summary(dialogue, max_length=max_length, min_length=min_length, do_sample=do_sample)

        # Split summary in list of sentences
        sentence_list = self.generate_list_of_sentences(summary[0]['summary_text'])

        return sentence_list

