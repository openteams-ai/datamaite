# Category taxonomies and ID density

## What a class id means

Every labeled box or image carries a class. A {term}`taxonomy` is the table that
gives class ids their meaning: id `1` is "car" in one dataset and "pedestrian"
in another. An id is only meaningful together with the taxonomy it came from.

## Source ids and dense ids

Datasets number their classes differently. Some use sparse or fixed numbers,
some use strings or synsets, and some have no ids at all. Some formats, such as
YOLO, need *dense* ids: contiguous integers starting at zero. This is
{term}`ID density`.

DataMAITE keeps each class's *source* id exactly as the dataset gave it, and
derives dense ids as a separate view when a format needs them, rather than
renumbering the stored classes. Classes are identified by their source dataset
together with their source id, so class `0` from two different datasets is never
silently merged into one class.

## Moving between class tables

Some formats have a fixed class table: MOTChallenge's classes and VisDrone's
categories, where for example VisDrone category `0` means "ignored region".
Writing into such a format is a translation between taxonomies. A
{term}`fixed-taxonomy writer` resolves each box's class in order of trust:

1. an explicit `class_map` you supply, by category name or id; categories it
   does not mention are dropped;
2. a class id the source format recorded for that target, when there is one;
3. the generic class id the loader assigned.

The last step is a {term}`generic-id reinterpretation`: a number that meant one
thing in the source is read as whatever the target table says that number
means. It can produce labels that look valid and are wrong, so DataMAITE warns
once per write whenever a generic id is written into the target table, and
whenever a `class_map` drops categories or maps two different source classes
onto one target class. Supplying a `class_map` makes the translation explicit.

## Further reading

- [Architecture](../reference/architecture.md): the `Taxonomy` model and the
  fixed-taxonomy class mapping.
