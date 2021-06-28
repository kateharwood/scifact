import sys
import spacy
import string
import editdistance

EN = spacy.load('en_core_web_trf')

VALID_MODES = [
	'binary', 'continuous'
]

# basically all non-numerical named entity types
VALID_ENTS = [
	'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 
	'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'
]


def lemmatize(text):
	return text.lower().translate(str.maketrans('', '', string.punctuation))


def get_constituents(sentence, include_nouns):
	parsed = EN(sentence)

	constituents = set()
	for ne in parsed.ents:
		if ne.label_ in VALID_ENTS:
			constituents.add(lemmatize(ne.lemma_))

	if include_nouns:
		for token in parsed:
			# it's a noun that's not part of a named entity
			if token.pos_ in ['PROPN', 'NOUN'] and token.ent_iob == 2:
				constituents.add(lemmatize(token.lemma_))

	return constituents


def get_specificity(claim, evidence, include_nouns, fuzzy, mode):
	claim_constituents = get_constituents(claim, include_nouns)

	if len(claim_constituents) == 0:
		return -1

	evidence_constituents = get_constituents(evidence, include_nouns)

	if not fuzzy:
		overlapping_constituents = claim_constituents & evidence_constituents
	else:
		overlapping_constituents = set()
		for claim_constituent in claim_constituents:
			for evidence_constituent in evidence_constituents:
				if claim_constituent == evidence_constituent:
					overlapping_constituents.add(claim_constituent)

				if len(claim_constituent) > 3 and len(evidence_constituent) > 3:
					if claim_constituent in evidence_constituent or evidence_constituent in claim_constituent:
						overlapping_constituents.add(claim_constituent)

					if editdistance.eval(claim_constituent, evidence_constituent) < len(claim_constituent)/2:
						overlapping_constituents.add(claim_constituent)

	if mode == 'binary':
		return len(overlapping_constituents) == len(claim_constituents)
	elif mode == 'continuous':
		return len(overlapping_constituents)/len(claim_constituents)
	else:
		raise Exception("Unsupported mode: %s" % mode)


if __name__ == '__main__':
	claim = sys.argv[1]
	evidence = sys.argv[2]

	print("claim: %s" % claim)
	print("evidence: %s" % evidence)

	for include_nouns in [True, False]:
		for fuzzy in [True, False]:
			print("\ninclude_nouns=%s, fuzzy=%s" % (include_nouns, fuzzy))
			print("claim constituents: %s" % (','.join(get_constituents(claim, include_nouns))))
			print("evidence constituents: %s" % (','.join(get_constituents(evidence, include_nouns))))
			print("specificity=%0.4f" % get_specificity(claim, evidence, include_nouns, fuzzy, 'continuous'))