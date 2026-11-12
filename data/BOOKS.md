# The six books

Everything our tiny AI knows comes from these six books. They were all published long ago, so they're in the public domain in the United States and anyone there can use them freely. If you live somewhere else, check your country's rules.

| Book | Author | Written | Source |
|---|---|---|---|
| Relativity: The Special and General Theory, translated by Robert W. Lawson | Albert Einstein | 1916 (this edition 1924) | [Project Gutenberg #5001](https://www.gutenberg.org/ebooks/5001) |
| Sidelights on Relativity | Albert Einstein | 1920-21 | [Project Gutenberg #7333](https://www.gutenberg.org/ebooks/7333) |
| On the Origin of Species, 1st edition | Charles Darwin | 1859 | [Project Gutenberg #1228](https://www.gutenberg.org/ebooks/1228) |
| The Chemical History of a Candle | Michael Faraday | 1860-61 lectures | [Project Gutenberg #14474](https://www.gutenberg.org/ebooks/14474) |
| The Story of the Heavens | Robert Stawell Ball | 1886 | [Project Gutenberg #27378](https://www.gutenberg.org/ebooks/27378) |
| Astronomy for Amateurs, translated by Frances A. Welby | Camille Flammarion | 1904 | [Project Gutenberg #25267](https://www.gutenberg.org/ebooks/25267) |

Thank you to the Project Gutenberg volunteers who typed up and proofread these books.

## How we cleaned them

[`corpus.txt`](corpus.txt) is all six books in one file, one paragraph per line. We:

- removed each book's Project Gutenberg header, footer and license text,
- removed illustration captions, footnote markers, headings, tables and equations,
- turned curly quotes and long dashes into plain ones,
- kept only plain letters, digits and basic punctuation, 75 different characters in all.

That leaves 2,612,504 characters, about 450,000 words. To train the AI in the video we held back every twentieth paragraph to test it on, so it learned from the other 95%. `tiny_ai.py` learns from all of it.
