from booknlp.booknlp import BookNLP
import os

# Setup BookNLP
model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "big"
}
booknlp = BookNLP("en", model_params)

# Input and output
input_file = "data/pride_and_prejudice_sample.txt"
output_directory = "results/booknlp_test"
book_id = "p_and_p_sample"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Run BookNLP
booknlp.process(input_file, output_directory, book_id)

print(f"BookNLP processing complete. Results in {output_directory}")
