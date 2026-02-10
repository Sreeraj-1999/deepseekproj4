from unstructured.partition.pdf import partition_pdf

fname = r"C:\4. Manuals\CD 01 Man B&W Diesel Engine\Marine Diesel Engine Manual.pdf"

elements = partition_pdf(filename=fname,
                         skip_infer_table_types=False,
                         strategy='hi_res',
           )

tables = [el for el in elements if el.category == "Table"]

print(tables[0].text)
print(tables[0].metadata.text_as_html)