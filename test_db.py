import asyncio
import asyncpg
import socket

async def test_database_connection():
    """Comprehensive database connection test"""
    
    # Test 1: Basic port connectivity
    print("=" * 50)
    print("🔧 Test 1: Port Connectivity")
    print("=" * 50)
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(("192.168.18.176", 5432))
        sock.close()
        if result == 0:
            print("✅ Port 5432 is OPEN on 192.168.18.176")
        else:
            print("❌ Port 5432 is CLOSED on 192.168.18.176")
            return
    except Exception as e:
        print(f"❌ Port test error: {e}")
        return
    
    # Test 2: Database connection with original password
    print("\n" + "=" * 50)
    print("🔧 Test 2: Original Connection String")
    print("=" * 50)
    
    DATABASE_URL_1 = "postgresql://memphis:Memphis@1234$@192.168.18.176/OBANEXT5"
    print(f"Testing: {DATABASE_URL_1}")
    
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(DATABASE_URL_1), 
            timeout=10.0
        )
        print("✅ Original connection string works!")
        await conn.close()
        return
    except asyncio.TimeoutError:
        print("❌ Connection timeout with original string")
    except Exception as e:
        print(f"❌ Original connection failed: {e}")
    
    # Test 3: Database connection with URL-encoded password
    print("\n" + "=" * 50)
    print("🔧 Test 3: URL-Encoded Password")
    print("=" * 50)
    
    DATABASE_URL_2 = "postgresql://memphis:Memphis%401234%24@192.168.18.176/OBANEXT5"
    print(f"Testing: {DATABASE_URL_2}")
    
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(DATABASE_URL_2), 
            timeout=10.0
        )
        print("✅ URL-encoded connection string works!")
        await conn.close()
        return
    except asyncio.TimeoutError:
        print("❌ Connection timeout with URL-encoded string")
    except Exception as e:
        print(f"❌ URL-encoded connection failed: {e}")
    
    # Test 4: Try different database name
    print("\n" + "=" * 50)
    print("🔧 Test 4: Try Default 'postgres' Database")
    print("=" * 50)
    
    DATABASE_URL_3 = "postgresql://memphis:Memphis%401234%24@192.168.18.176/postgres"
    print(f"Testing: {DATABASE_URL_3}")
    
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(DATABASE_URL_3), 
            timeout=10.0
        )
        print("✅ Connection to 'postgres' database works!")
        print("ℹ️  Issue might be that 'OBANEXT5' database doesn't exist")
        await conn.close()
        return
    except asyncio.TimeoutError:
        print("❌ Connection timeout with postgres database")
    except Exception as e:
        print(f"❌ Postgres database connection failed: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 Summary")
    print("=" * 50)
    print("All connection attempts failed.")
    print("Possible issues:")
    print("1. Username 'memphis' doesn't exist")
    print("2. Password is incorrect")
    print("3. Database 'OBANEXT5' doesn't exist")
    print("4. User doesn't have permission to connect")
    print("5. PostgreSQL authentication configuration issue")

if __name__ == "__main__":
    print("🚀 Starting comprehensive database connection test...")
    asyncio.run(test_database_connection())