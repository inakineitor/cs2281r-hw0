from pathlib import Path
import pypandoc
import glob
import typer


def main(file_directory: Path, output_file_path: Path):
    print(pypandoc.get_pandoc_formats()[1])
    file_glob_pattern = str(file_directory / "**.*")
    print(f"Looking for files in {file_glob_pattern}")

    matched_file_paths = sorted(glob.glob(file_glob_pattern))

    output_file_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure parent directories of output file exist

    with open(str(output_file_path), "w", encoding="utf-8") as output_file:
        for input_file_path in matched_file_paths:
            print(f"Converting {input_file_path} ...")
            document_contents = pypandoc.convert_file(
                input_file_path,
                "plain",
            )
            output_file.write("\n\n")
            output_file.write(document_contents)

    print("Processing finished :)")


if __name__ == "__main__":
    typer.run(main)
