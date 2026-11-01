# majorproject

// 1. Load mentions from gold.mentions (Component 2); unlinked spans are kept
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
LOAD CSV WITH HEADERS FROM 'file:///gold_mentions.csv' AS r
MERGE (t:Tweet {id: r.tweet_id})
  SET t.retweets = toInteger(r.retweets), t.favourites = toInteger(r.favourites)
MERGE (e:Entity {id: coalesce(r.qid, 'span:' + toLower(r.span))})
  ON CREATE SET e.name = r.span, e.qid = r.qid, e.type = r.ent_type,
                e.linked = r.qid IS NOT NULL
MERGE (t)-[:MENTIONS]->(e);
MATCH (e:Entity) WHERE e.type = 'HASHTAG' SET e:Hashtag;
MATCH (e:Entity) WHERE e.type = 'HANDLE'  SET e:Handle;
 
// 2. Derive weighted co-mention edges
MATCH (e1:Entity)<-[:MENTIONS]-(t:Tweet)-[:MENTIONS]->(e2:Entity) WHERE e1.id < e2.id
WITH e1, e2, count(DISTINCT t) AS w WHERE w >= 3
MERGE (e1)-[c:CO_MENTIONED]-(e2) SET c.weight = w;
 
// 3. Graph algorithms
// Collection keywords (Olympics, #tokyo2020 ...) occur in almost every tweet
MATCH (a:Entity)-[c:CO_MENTIONED]-(b:Entity)
WHERE NOT a.name IN $collection_terms AND NOT b.name IN $collection_terms
WITH gds.graph.project('ent', a, b, {relationshipProperties: c {.weight}},
       {undirectedRelationshipTypes: ['*']}) AS g
RETURN g.nodeCount, g.relationshipCount;
CALL gds.louvain.write('ent',{relationshipWeightProperty:'weight',
  writeProperty:'community'}) YIELD communityCount, modularity;
CALL gds.pageRank.write('ent',{relationshipWeightProperty:'weight',
  writeProperty:'pagerank'});
CALL gds.betweenness.write('ent',{writeProperty:'betweenness'});
 
// 4. Evaluation: amplification per community (each tweet counted once)
MATCH (e:Entity)<-[:MENTIONS]-(t:Tweet)
WITH DISTINCT e.community AS c, t
WITH c, count(t) AS n,
     avg(CASE WHEN t.retweets > 0 THEN 1.0 ELSE 0.0 END) AS share_retweeted,
     percentileCont(t.retweets, 0.99) AS p99
RETURN c, n, round(share_retweeted, 3) AS share_retweeted, p99
ORDER BY share_retweeted DESC LIMIT 10;
